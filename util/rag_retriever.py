"""
RAG retrievalUtilitymodule

RustEvo.json docsretrieval, model API info

implementmode:
1. Simple mode(, recommended):directly RustEvo.json retrieval, dependencies
2. Vector mode():Use langchain + chroma Buildvectorindex

Usage:
    from util.rag_retriever import get_rag_documents
    
    docs = get_rag_documents(
        query="how to open a file", 
        api_name="open", 
        module="std::fs::File",
        top_k=3
    )
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()


class SimpleRustEvoRetriever:
    """ APIDocs.json retrieval(vectordata, without answer code)"""
    
    def __init__(
        self,
        data_path: str = "data/RustEvo/APIDocs.json",
        top_k: int = 3
    ):
        """
        retrieval
        
        Args:
            data_path: APIDocs.json filepath(clean API docs, without)
            top_k: returnNumber of documents
        """
        self.data_path = Path(data_path)
        self.top_k = top_k
        self.api_data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """load RustEvo.json data"""
        if not self.data_path.exists():
            print(f"Warning: {self.data_path} not found")
            return []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def retrieve(
        self,
        query: str,
        api_name: Optional[str] = None,
        module: Optional[str] = None,
        crate_name: Optional[str] = None
    ) -> str:
        """
        retrievaldocs
        
        strategy:
        1. match:name + module
        2. namematch:match name
        3. modulematch: module
        4. match: query extract
        """
        if not self.api_data:
            return "No documentation available."
        
        if api_name and module:
            exact_matches = [item for item in self.api_data 
                           if item.get("name") == api_name and item.get("module") == module]
            if exact_matches:
                return self._format_documents(exact_matches[:self.top_k])
        
        if api_name:
            name_matches = [item for item in self.api_data if item.get("name") == api_name]
            if name_matches:
                return self._format_documents(name_matches[:self.top_k])
        
        if module:
            module_prefix = module.split("::")[0]
            module_matches = [item for item in self.api_data 
                            if item.get("module", "").startswith(module_prefix)]
            if module_matches:
                return self._format_documents(module_matches[:self.top_k])
        
        if query:
            keywords = self._extract_keywords(query)
            scored_items = []
            for item in self.api_data:
                score = self._compute_relevance(item, keywords)
                if score > 0:
                    scored_items.append((score, item))
            
            scored_items.sort(reverse=True, key=lambda x: x[0])
            top_items = [item for _, item in scored_items[:self.top_k]]
            if top_items:
                return self._format_documents(top_items)
        
        return "No relevant documentation found."
    
    def _extract_keywords(self, query: str) -> List[str]:
        """ query extract"""
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 
                     'at', 'to', 'for', 'of', 'with', 'by', 'from', 'how', 'can',
                     'you', 'i', 'we', 'they', 'that', 'this', 'what', 'when'}
        words = query.lower().split()
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _compute_relevance(self, item: Dict, keywords: List[str]) -> float:
        """docs"""
        text = f"{item.get('name', '')} {item.get('module', '')} {item.get('documentation', '')} {item.get('query', '')}"
        text_lower = text.lower()
        return sum(1.0 for keyword in keywords if keyword in text_lower)
    
    def _format_documents(self, items: List[Dict]) -> str:
        """docs"""
        if not items:
            return "No documentation found."
        
        formatted = []
        for i, item in enumerate(items, 1):
            doc = f"[Document {i}]\n"
            doc += f"API: {item.get('name', 'N/A')}\n"
            doc += f"Module: {item.get('module', 'N/A')}\n"
            doc += f"Type: {item.get('type', 'N/A')}\n"
            
            if sig := item.get('signature'):
                doc += f"Signature: {sig}\n"
            
            if documentation := item.get('documentation'):
                doc += f"Documentation: {documentation}\n"
            
            if source := item.get('source_code'):
                if len(source) < 500:
                    doc += f"Source Code:\n{source}\n"
            
            from_ver = item.get('from_version', '')
            to_ver = item.get('to_version', '')
            if from_ver and to_ver:
                doc += f"Version: {from_ver} → {to_ver}\n"
            
            formatted.append(doc)
        
        return "\n".join(formatted)


class VectorRustEvoRetriever:
    """vectordataretrieval( play implement, Useclean API docs)"""
    
    def __init__(
        self,
        data_path: str = "data/RustEvo/APIDocs.json",
        embedding_model: str = "OpenAI",
        persist_directory: str = ".rag_cache/chroma_db",
        top_k: int = 3,
        openai_api_key: str = None,
        openai_base_url: str = None
    ):
        """
        vectorretrieval
        
        Args:
            data_path: APIDocs.json filepath(clean API docs, without)
            embedding_model: "OpenAI" "HuggingFace"
            persist_directory: vectordataSavedirectory
            top_k: returnNumber of documents
            openai_api_key: OpenAI API Key(, environmentread)
            openai_base_url: OpenAI Base URL(, API)
        """
        self.data_path = Path(data_path)
        self.top_k = top_k
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        
        try:
            from langchain_community.document_loaders import JSONLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_chroma import Chroma
            self.available = True
        except ImportError:
            print("Error: langchain not installed.")
            print("Install with: pip install langchain langchain-community langchain-chroma langchain-openai")
            self.available = False
            return
        
        self.embeddings = self._init_embeddings()
        self.vector_store = self._init_vector_store()
    
    def _init_embeddings(self):
        """ embedding model"""
        if self.embedding_model_name == "OpenAI":
            from langchain_openai import OpenAIEmbeddings
            api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set. Please provide openai_api_key parameter or set OPENAI_API_KEY environment variable.")
            
            kwargs = {"model": "text-embedding-3-large"}
            if api_key:
                kwargs["openai_api_key"] = api_key
            if self.openai_base_url:
                kwargs["openai_api_base"] = self.openai_base_url
            
            return OpenAIEmbeddings(**kwargs)
        elif self.embedding_model_name == "HuggingFace":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        else:
            raise ValueError(f"Unknown embedding model: {self.embedding_model_name}")
    
    def _init_vector_store(self):
        """loadvectordata( RustEvo/RAG_unit.py)"""
        from langchain_chroma import Chroma

        persist_path = Path(self.persist_directory)

        if persist_path.exists() and (persist_path / "chroma.sqlite3").exists():
            print(f"Loading existing vector store from {self.persist_directory}")
            vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            existing_ids = vector_store.get()["ids"]
            if existing_ids:
                print(f"Loaded existing store with {len(existing_ids)} documents")
                return vector_store

        print(f"Building new vector store from {self.data_path}")
        docs = self._load_and_split_documents()

        if not docs:
            raise ValueError("No documents to index")

        persist_path.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        batch_size = 100
        total_docs = len(docs)
        print(f"Embedding {total_docs} document chunks...")

        for i in range(0, total_docs, batch_size):
            batch = docs[i:i+batch_size]
            vector_store.add_documents(documents=batch)
            print(f"Processed {min(i+batch_size, total_docs)}/{total_docs} chunks")

        print(f"Vector store built successfully")
        return vector_store
    
    def _load_and_split_documents(self):
        """loaddocs( RustEvo/RAG_unit.py)"""
        from langchain_community.document_loaders import JSONLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if not self.data_path.exists():
            print(f"Warning: {self.data_path} not found")
            return []

        def metadata_func(record: dict, metadata: dict) -> dict:
            if "crate" not in record:
                metadata["name"] = record.get("name")
                metadata["from_version"] = record.get("from_version")
                metadata["to_version"] = record.get("to_version")
                metadata["module"] = record.get("module")
                metadata["type"] = record.get("type")
                metadata["signature"] = record.get("signature")
                metadata["documentation"] = record.get("documentation")
                metadata["source_code"] = record.get("source_code")
            else:
                metadata["crate"] = record.get("crate")
                metadata["name"] = record.get("name")
                metadata["from_version"] = record.get("from_version")
                metadata["to_version"] = record.get("to_version")
                metadata["module"] = record.get("module")
                metadata["type"] = record.get("type")
                metadata["signature"] = record.get("signature")
                metadata["documentation"] = record.get("documentation")
                metadata["source_code"] = record.get("source_code")

            for key, value in list(metadata.items()):
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
                elif value is None:
                    metadata[key] = ""
            return metadata

        loader = JSONLoader(
            file_path=str(self.data_path),
            jq_schema=".[]",
            metadata_func=metadata_func,
            text_content=False
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        return text_splitter.split_documents(docs)
    
    def retrieve(
        self,
        query: str,
        api_name: Optional[str] = None,
        module: Optional[str] = None,
        crate_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """vectorretrieval"""
        if not self.available:
            return "Vector RAG not available. Please install langchain."
        
        enhanced_query = query
        if api_name:
            enhanced_query += f" API: {api_name}"
        if module:
            enhanced_query += f" Module: {module}"
        
        retrieved_docs = self.vector_store.similarity_search(
            enhanced_query,
            k=self.top_k
        )
        
        if not retrieved_docs:
            return "No relevant documentation found."
        
        formatted = []
        for i, doc in enumerate(retrieved_docs, 1):
            formatted_doc = f"[Document {i}]\n"
            formatted_doc += f"{doc.page_content}\n"
            
            if doc.metadata:
                formatted_doc += "---\n"
                for key, value in doc.metadata.items():
                    if value:
                        formatted_doc += f"{key}: {value}\n"
            
            formatted.append(formatted_doc)
        
        return "\n".join(formatted)


def get_rag_documents(
    query: str,
    api_name: Optional[str] = None,
    module: Optional[str] = None,
    crate_name: Optional[str] = None,
    top_k: int = 3,
    data_path: str = "data/RustEvo/APIDocs.json",
    use_vector: bool = False,
    embedding_model: str = "OpenAI",
    openai_api_key: str = None,
    openai_base_url: str = None
) -> str:
    """
    retrievaldocs(clean API docsretrieval, without answer code)
    
    Args:
        query: query
        api_name: API name
        module: modulepath
        crate_name: Crate name(Simple mode)
        top_k: returnNumber of documents
        data_path: RustEvo.json path
        use_vector: Usevectorretrieval( langchain + API key)
        embedding_model: "OpenAI" "HuggingFace"
        openai_api_key: OpenAI API Key(, Usecodeconfig)
        openai_base_url: OpenAI Base URL(, Usecodeconfig)
        
    Returns:
        Retrieveddocs
    """
    if openai_api_key is None and OPENAI_API_KEY:
        openai_api_key = OPENAI_API_KEY
    if openai_base_url is None and OPENAI_BASE_URL:
        openai_base_url = OPENAI_BASE_URL
    
    if use_vector:
        retriever = VectorRustEvoRetriever(
            data_path=data_path,
            embedding_model=embedding_model,
            persist_directory=".rag_cache/chroma_db",
            top_k=top_k,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url
        )
    else:
        retriever = SimpleRustEvoRetriever(data_path=data_path, top_k=top_k)
    
    return retriever.retrieve(query, api_name, module, crate_name)


def get_RAG_document(query: str, use_vector: bool = False) -> str:
    """( play code)"""
    return get_rag_documents(query, top_k=3, use_vector=use_vector)


if __name__ == "__main__":
    print("Testing RAG retriever...")
    
    print("\nTest 1: Simple retrieval")
    docs = get_rag_documents(
        query="How to change file ownership",
        api_name="chown",
        module="std::os::unix::fs",
        use_vector=False
    )
    print(docs[:500])
    
    print("\nTest 2: Vector retrieval (needs OPENAI_API_KEY)")
    try:
        docs = get_rag_documents(
            query="How to divide integers and round up",
            use_vector=True,
            embedding_model="OpenAI",
            top_k=2
        )
        print(docs[:800])
    except Exception as e:
        print(f"Vector retrieval failed: {e}")
        print("Make sure: 1) OPENAI_API_KEY is set, 2) pip install langchain langchain-openai langchain-community langchain-chroma")
