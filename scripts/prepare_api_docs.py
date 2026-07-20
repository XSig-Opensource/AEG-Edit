"""
Extract clean API documentation from RustEvo.json for RAG retrieval

Keep only API info, remove answer-related fields:
- : name, module, type, signature, documentation, source_code, examples, versions
- : code, test_program, function_signature, query, rephrased_query
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def extract_api_docs(rustevo_data: List[Dict]) -> List[Dict]:
    """Extract clean API documentation from RustEvo.json"""
    api_docs = []
    seen_apis = set()
    
    for item in rustevo_data:
        api_key = (item.get('name', ''), item.get('module', ''), item.get('to_version', ''))
        
        if api_key in seen_apis:
            continue
        seen_apis.add(api_key)
        
        api_doc = {
            'name': item.get('name', ''),
            'module': item.get('module', ''),
            'type': item.get('type', ''),
            'signature': item.get('signature', ''),
            'documentation': item.get('documentation', ''),
            'source_code': item.get('source_code', ''),
            'from_version': item.get('from_version', ''),
            'to_version': item.get('to_version', ''),
            'change_type': item.get('change_type', ''),
        }
        
        if 'examples' in item and item['examples']:
            api_doc['examples'] = item['examples']
        
        if 'old_source_code' in item and item['old_source_code']:
            api_doc['old_source_code'] = item['old_source_code']
        
        if 'crate' in item:
            api_doc['crate'] = item['crate']
        
        api_docs.append(api_doc)
    
    return api_docs


def main():
    parser = argparse.ArgumentParser(description='Generate clean API documentation from RustEvo.json')
    parser.add_argument('--input', type=str, default='data/RustEvo/RustEvo.json',
                        help='Input RustEvo.json path')
    parser.add_argument('--output', type=str, default='data/RustEvo/APIDocs.json',
                        help='Output APIDocs.json path')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}")
        return
    
    print(f"Read RustEvo data: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        rustevo_data = json.load(f)
    
    print(f"Original data entries: {len(rustevo_data)}")
    
    api_docs = extract_api_docs(rustevo_data)
    
    print(f"Extracted API docs: {len(api_docs)} (deduplicated)")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(api_docs, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_path}")
    
    print("\nStatistics:")
    print(f" - Total APIs: {len(api_docs)}")
    
    type_count = {}
    for doc in api_docs:
        t = doc.get('type', 'unknown')
        type_count[t] = type_count.get(t, 0) + 1
    
    print(" - by type:")
    for t, count in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"      {t}: {count}")
    
    module_count = {}
    for doc in api_docs:
        module = doc.get('module', '')
        top_module = module.split('::')[0] if module else 'unknown'
        module_count[top_module] = module_count.get(top_module, 0) + 1
    
    print(" - Distribution by top-level module:")
    for m, count in sorted(module_count.items(), key=lambda x: -x[1])[:10]:
        print(f"      {m}: {count}")
    
    empty_docs = sum(1 for doc in api_docs if not doc.get('documentation'))
    empty_source = sum(1 for doc in api_docs if not doc.get('source_code'))
    print(f"\n - Missing documentation: {empty_docs}")
    print(f" - Missing source_code: {empty_source}")
    
    print("\n[OK] Done! APIDocs.json is ready for RAG retrieval")


if __name__ == "__main__":
    main()
