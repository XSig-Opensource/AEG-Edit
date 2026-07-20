"""
BaselineevaluateScript - RAG (w/ RAG only)

Use RAG retrievalreplacementDetailed API info, evaluateretrievalgenerate

Support RAG mode:
- Simple mode(): RustEvo.json directlyretrieval, dependencies
- Vector mode(--use_vector):Use langchain + chroma vectorretrieval

Prompt content (RAG only):
- Task Description (query)
- Required Function Signature (function_signature)
- API Name + Module (Minimal information)
- **Retrieved Documentation (RAG retrieval)** ← Core

Note:RAG retrievalUse APIDocs.json(clean API docs, without answer code)

Validateworkflow:
- Check function signature match
- Check API usage correctness
- Run Rust tests
"""

import json
import os
import re
import argparse
import numpy as np
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import httpx
import sys
from pathlib import Path
from util.openai_runtime import get_openai_runtime_config

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from util.rust_cargo_test import (
    run_rust_test_auto,
    check_function_signature,
    check_api_usage,
    prepare_rust_test_inputs,
    DEFAULT_TEST_TIMEOUT
)
from util.rag_retriever import get_rag_documents


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Pass@k Metrics

    Args:
        n: Total samples
        c: correctsample
        k: k in pass@k

    Returns:
        pass@k (0-1)

    Formula: 1 - C(n-c, k) / C(n, k)
    """
    if n == 0:
        return 0.0
    if c > n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


OPENAI_API_KEY, OPENAI_BASE_URL, _ = get_openai_runtime_config("deepseek-v3.2")
#==

 
def call_LLM(prompt: str, model: str, base_url: str, max_tokens: int = 512) -> str:
    """callLLM APIreturnresponse"""
    client = OpenAI(api_key="EMPTY", base_url=base_url, http_client=httpx.Client(trust_env=False))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.001,
            max_tokens=max_tokens
        )
        code = response.choices[0].message.content.strip()
        return code
    except Exception as e:
        print(f"Error calling LLM {model}: {str(e)}")
        return ""


def build_prompt_with_rag(sample: dict, rag_top_k: int = 3, use_vector: bool = False, embedding_model: str = "OpenAI", openai_api_key: str = None, openai_base_url: str = None, rag_path: str = "data/RustEvo/APIDocs.json") -> str:
    """Build RAG prompt( play , API info + RAG)"""
    query = sample.get("query", "")
    function_signature = sample.get("function_signature", "")
    name = sample.get("name", "")
    module = sample.get("module", "")
    to_version = sample.get("to_version", "1.84.0")
    
    is_crate = not (module.startswith("std::") or module.startswith("core::") or module.startswith("alloc::"))
    crate_name = None
    if is_crate:
        crate_name = module.split("::")[0] if "::" in module else module
    
    retrieved_docs = get_rag_documents(
        query=query,
        api_name=name,
        module=module,
        crate_name=crate_name,
        top_k=rag_top_k,
        data_path=rag_path,
        use_vector=use_vector,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url
    )
    
    if not is_crate:
        prompt = f"""
    You are an expert Rust programmer. Implement the following Rust function:

    API Information: 
    - API Name: {name}
    - API Module: {module}

    Task: {query}
    Function Signature:
    ```rust
    {function_signature}
    ```

    Retrieved Documentation:
    {retrieved_docs}

    Requirements:
    1. Implement ONLY the Rust function with the signature given above.
    2. Your implementation MUST use the specified API: {name}
    3. Make sure your code is compatible with Rust version {to_version}
    4. Do not include tests or any extra comments.

    Respond with ONLY the Rust function implementation.
"""
    else:
        prompt = f"""
    You are an expert Rust programmer. Implement the following Rust function:

    API Information:
    - Crate Name: {crate_name}
    - API Name: {name}
    - API Module: {module}
    
    Task: {query}
    Function Signature:
    ```rust
    {function_signature}
    ```

    Retrieved Documentation:
    {retrieved_docs}

    Requirements:
    1. Implement ONLY the Rust function with the signature given above.
    2. Your implementation MUST use the specified API: {name}
    3. Compile with Rust 1.84.0. Do not include tests or any extra comments.

    Respond with ONLY the Rust function implementation.
"""
    
    return prompt


def extract_rust_code(response: str) -> str:
    """LLMresponseextractRustcode"""
    rust_pattern = r"```(?:rust)?\s*([\s\S]*?)```"
    matches = re.findall(rust_pattern, response)
    
    if matches:
        valid_matches = [m.strip() for m in matches if m.strip()]
        if valid_matches:
            return max(valid_matches, key=len)
    
    lines = response.strip().split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        if re.match(r'\s*(use|fn|pub|struct|impl|mod)\s+', line):
            in_code_block = True
        
        if in_code_block:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else response


def process_sample(
    sample: dict,
    sample_idx: int,
    model: str,
    base_url: str,
    timeout: int = 30,
    max_tokens: int = 512,
    rag_top_k: int = 3,
    use_vector: bool = False,
    embedding_model: str = "OpenAI",
    openai_api_key: str = None,
    openai_base_url: str = None,
    rag_path: str = "data/RustEvo/APIDocs.json"
) -> dict:
    """processsample( RAG )"""
    sample_id = f"{sample_idx}_{sample.get('name', 'unknown')}"
    
    query = sample.get('query', '')
    test_program = sample.get('test_program', '')
    function_signature = sample.get('function_signature', '')
    name = sample.get('name', '')
    change_type = sample.get('change_type', '')
    api_module = sample.get('module', '')
    module = sample.get('module', '')
    to_version = sample.get('to_version', '1.84.0')
    
    if not query:
        return {
            'id': sample_id,
            'passed': False,
            'error': 'No query field',
            'generated_code': '',
            'model_response': '',
            'retrieved_docs': ''
        }
    
    if not test_program:
        return {
            'id': sample_id,
            'passed': False,
            'error': 'No test program',
            'generated_code': '',
            'model_response': '',
            'retrieved_docs': ''
        }
    
    prompt = build_prompt_with_rag(sample, rag_top_k=rag_top_k, use_vector=use_vector, embedding_model=embedding_model, openai_api_key=openai_api_key, openai_base_url=openai_base_url, rag_path=rag_path)
    
    is_crate = not (module.startswith("std::") or module.startswith("core::") or module.startswith("alloc::"))
    crate_name = None
    if is_crate:
        crate_name = module.split("::")[0] if "::" in module else module
    
    retrieved_docs = get_rag_documents(
        query=query,
        api_name=name,
        module=module,
        crate_name=crate_name,
        top_k=rag_top_k,
        data_path=rag_path,
        use_vector=use_vector,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url
    )
    
    model_response = call_LLM(prompt, model, base_url, max_tokens=max_tokens)
    
    if not model_response:
        return {
            'id': sample_id,
            'passed': False,
            'error': 'Empty LLM response',
            'generated_code': '',
            'model_response': '',
            'retrieved_docs': retrieved_docs
        }
    
    generated_code = extract_rust_code(model_response)
    
    if not generated_code:
        return {
            'id': sample_id,
            'passed': False,
            'error': 'Failed to extract Rust code',
            'generated_code': '',
            'model_response': model_response,
            'retrieved_docs': retrieved_docs
        }
    
    if function_signature:
        if not check_function_signature(generated_code, function_signature):
            return {
                'id': sample_id,
                'passed': False,
                'error': 'Incorrect function signature',
                'generated_code': generated_code,
                'model_response': model_response,
                'expected_signature': function_signature,
                'retrieved_docs': retrieved_docs
            }
    
    if name:
        replacement_api = sample.get('replacement_api', '')
        if not check_api_usage(generated_code, name, change_type, api_module, test_program, replacement_api):
            error_msg = (
                f'Deprecated API used: {name}'
                if str(change_type).lower() == 'deprecated'
                else f'Required API not used: {name}'
            )
            return {
                'id': sample_id,
                'passed': False,
                'error': error_msg,
                'generated_code': generated_code,
                'model_response': model_response,
                'expected_api': name,
                'retrieved_docs': retrieved_docs
            }
    
    clean_code, clean_test, rust_version, crate_version, _, _ = prepare_rust_test_inputs(
        generated_code,
        test_program,
        module,
        to_version,
        dedup_imports=True,
    )
    
    test_result = run_rust_test_auto(
        clean_code,
        clean_test,
        module,
        rust_version,
        crate_version,
        timeout=timeout
    )
    
    return {
        'id': sample_id,
        'passed': test_result['success'],
        'error': test_result.get('error', ''),
        'generated_code': generated_code,
        'model_response': model_response,
        'rust_version': rust_version,
        'retrieved_docs': retrieved_docs,
        'test_stdout': test_result.get('stdout', ''),
        'test_stderr': test_result.get('stderr', '')
    }


def main():
    parser = argparse.ArgumentParser(description='Baselineevaluate ( RAG , Useclean API docs)')
    parser.add_argument('--data_path', type=str, default='data/RustEvo/RustEvo.json',
                        help='RustEvodataset path(evaluate, RAG autoUse APIDocs.json)')
    parser.add_argument('--output', type=str, default='results/baseline_w_rag_results.json',
                        help='output results path')
    parser.add_argument('--model', type=str, default='Qwen2.5-7B-Instruct',
                        help='modelname')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000/v1',
                        help='vLLM API')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='worker')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TEST_TIMEOUT,
                        help='Timeout per sample (seconds)')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='LLMgeneratetoken')
    parser.add_argument('--rag_top_k', type=int, default=3,
                        help='RAG retrievalNumber of documents')
    parser.add_argument('--use_vector', action='store_true',
                        help='Usevectorretrieval( langchain + OPENAI_API_KEY)')
    parser.add_argument('--embedding_model', type=str, default='OpenAI',
                        choices=['OpenAI', 'HuggingFace'],
                        help='vectorretrieval embedding model')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API Key(, Usecodeconfig)')
    parser.add_argument('--openai_base_url', type=str, default=None,
                        help='OpenAI Base URL(, Usecodeconfig)')
    parser.add_argument('--start_id', type=int, default=None,
                        help='sampleID()')
    parser.add_argument('--end_id', type=int, default=None,
                        help='sampleID()')
    parser.add_argument('--rag_path', type=str, default='data/RustEvo/APIDocs.json',
                        help='RAG retrievaldocspath(APIDocs.json)')
    parser.add_argument('--test_python', type=str, default=None,
                        help='RusttestPythonpath(parameter, Use)')
    
    args = parser.parse_args()
    
    if args.openai_api_key is None and OPENAI_API_KEY and OPENAI_API_KEY != "sk-REPLACE_ME":
        args.openai_api_key = OPENAI_API_KEY
    if args.openai_base_url is None and OPENAI_BASE_URL:
        args.openai_base_url = OPENAI_BASE_URL
    
    print(f"Load dataset: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.start_id is not None or args.end_id is not None:
        start = args.start_id if args.start_id is not None else 0
        end = args.end_id if args.end_id is not None else len(data) - 1
        data = [s for i, s in enumerate(data) if start <= i <= end]
        print(f"filtersamplerange: index {start} {end}")
    
    print(f"Total samples: {len(data)}")
    print(f"model: {args.model}")
    print(f"API: {args.base_url}")
    print(f"RAG mode: {'vectorretrieval' if args.use_vector else 'retrieval'}")
    if args.use_vector:
        print(f"Embedding model: {args.embedding_model}")
    print(f"RAG top_k: {args.rag_top_k}")
    print(f": {args.max_workers}")
    
    results = []
    processed_ids = set()
    
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_ids = {r['id'] for r in results}
                print(f"loadresults: {len(results)}sample")
        except:
            print("loadresults, ")
    
    remaining_samples = [s for i, s in enumerate(data) if f"{i}_{s.get('name', 'unknown')}" not in processed_ids]
    print(f"processsample: {len(remaining_samples)}")
    
    if not remaining_samples:
        print("sampleProcessing complete!")
        return
    
    passed_count = 0
    failed_count = 0
    
    sample_to_idx = {id(s): i for i, s in enumerate(data)}
    
    with tqdm(total=len(remaining_samples), desc="evaluate") as pbar:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_sample,
                    sample,
                    sample_to_idx[id(sample)],
                    args.model,
                    args.base_url,
                    args.timeout,
                    args.max_tokens,
                    args.rag_top_k,
                    args.use_vector,
                    args.embedding_model,
                    args.openai_api_key,
                    args.openai_base_url,
                    args.rag_path
                ): sample
                for sample in remaining_samples
            }
            
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['passed']:
                        passed_count += 1
                    else:
                        failed_count += 1
                    
                    if len(results) % 10 == 0:
                        os.makedirs(os.path.dirname(args.output), exist_ok=True)
                        with open(args.output, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                
                except Exception as e:
                    print(f"\nprocesssample: {str(e)}")
                    failed_count += 1
                
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        'passed': passed_count,
                        'failed': failed_count,
                        'rate': f"{passed_count/(passed_count+failed_count)*100:.1f}%" if (passed_count+failed_count) > 0 else "0%"
                    })
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Aggregate results
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    
    sig_errors = sum(1 for r in results if not r['passed'] and 'signature' in r.get('error', '').lower())
    api_errors = sum(1 for r in results if not r['passed'] and 'api' in r.get('error', '').lower())
    extract_errors = sum(1 for r in results if not r['passed'] and 'extract' in r.get('error', '').lower())
    test_errors = failed - sig_errors - api_errors - extract_errors
    
    print("\n" + "="*70)
    print("Evaluation complete!( RAG - withoutDetailed API info)")
    print("="*70)
    print(f"Total samples: {total}")
    
    if total > 0:
        print("\n" + "─"*50)
        print("[Stats] RustEvo Core Metrics (RAG)")
        print("─"*50)
        
        for k in [1, 2, 5]:
            if k > total:
                continue
            pass_k = pass_at_k(total, passed, k) * 100
            print(f"  pass@{k}:  {pass_k:6.2f}%")
        
        print(f"\n[PASS] passed: {passed}/{total} ({passed/total*100:.2f}%)")
    
    if failed > 0:
        print(f"\n[FAIL] failed: {failed}/{total} ({failed/total*100:.2f}%)")
        print(f" - Function signature error: {sig_errors} ({sig_errors/total*100:.2f}%)")
        print(f" - APIUseerror: {api_errors} ({api_errors/total*100:.2f}%)")
        print(f" - Code extraction failed: {extract_errors} ({extract_errors/total*100:.2f}%)")
        print(f" - testfailed: {test_errors} ({test_errors/total*100:.2f}%)")
    
    print(f"\nResults saved to: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()
