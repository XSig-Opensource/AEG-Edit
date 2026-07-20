"""
BaselineevaluateScript - w/o API (without detailed API information)

corresponding PyEvo project:LLM API namemodule Python codegeneratecapability

Prompt content (w/o API):
- Task Description (query)
- Required Function Signature (function_signature)
- Relevant API Information:
  * API Name (onlyname)
  * API Module (onlymodulepath)
  * [FAIL] without:signature, docs, source code, versioninfo

Validateworkflow:
- Check function signature match
- Check API usage correctness
- Run Python pytest tests
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

# Force bypass proxy for localhost connections
os.environ.setdefault('no_proxy', 'localhost,127.0.0.1')
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')

# Insert util/ directly so we avoid util/__init__.py (which imports torch)
_UTIL_DIR = str(Path(__file__).parent.parent.parent / "util")
if _UTIL_DIR not in sys.path:
    sys.path.insert(0, _UTIL_DIR)

from python_pytest_test import (
    run_python_test,
    check_function_signature,
    check_api_usage,
    extract_python_code,
    prepare_test_inputs,
    DEFAULT_TEST_TIMEOUT,
    DEFAULT_PYTHON_BIN,
    get_sandbox_python,
)


def _parse_test_cases(output: str):
    """ pytest OutputParse passed/total test case """
    m_p = re.search(r'(\d+) passed', output)
    m_f = re.search(r'(\d+) failed', output)
    m_e = re.search(r'(\d+) error', output)
    passed = int(m_p.group(1)) if m_p else 0
    failed = int(m_f.group(1)) if m_f else 0
    errors = int(m_e.group(1)) if m_e else 0
    return passed, passed + failed + errors


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


def call_LLM(prompt: str, model: str, base_url: str, max_tokens: int = 512) -> str:
    """callLLM APIreturnresponse"""
    http_client = httpx.Client(trust_env=False)
    client = OpenAI(api_key="EMPTY", base_url=base_url, http_client=http_client)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.001,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM {model}: {str(e)}")
        return ""


def get_code_generation_prompt(sample: dict) -> str:
    """Buildcodegenerateprompt - w/o API (onlynamemodule)"""
    query = sample.get("query", "")
    function_signature = sample.get("function_signature", "")
    name = sample.get("name", "")
    module = sample.get("module", "")

    prompt = f"""You are an expert Python programmer. Implement the following Python function.

API Information:
- API Name: {name}
- API Module: {module}

Task: {query}

Function Signature:
```python
{function_signature}
```

Requirements:
1. Implement ONLY the function with the signature given above.
2. Your implementation MUST use the specified API: {name}
3. Do not include tests or any extra comments.

Respond with ONLY the Python function implementation.
"""
    return prompt


def process_sample(sample: dict, sample_idx: int, model: str, base_url: str,
                   timeout: int = 60, max_tokens: int = 512,
                   test_python: str = None,
                   use_sandbox: bool = False,
                   sandbox_cache: str = None) -> dict:
    """processsample(testversionSandbox environment)"""
    sample_id = f"{sample_idx}_{sample.get('name', 'unknown')}"

    query = sample.get("query", "")
    test_program = sample.get("test_program", "")
    function_signature = sample.get("function_signature", "")
    name = sample.get("name", "")
    change_type = sample.get("change_type", "")
    module = sample.get("module", "")
    to_version = sample.get("to_version", "")
    if not query:
        return {
            "id": sample_id, "passed": False, "error": "No query field",
            "generated_code": "", "model_response": "",
        }

    if not test_program:
        return {
            "id": sample_id, "passed": False, "error": "No test program",
            "generated_code": "", "model_response": "",
        }

    prompt = get_code_generation_prompt(sample)
    model_response = call_LLM(prompt, model, base_url, max_tokens=max_tokens)

    if not model_response:
        return {
            "id": sample_id, "passed": False, "error": "Empty LLM response",
            "generated_code": "", "model_response": "",
        }

    generated_code = extract_python_code(model_response)

    if not generated_code:
        return {
            "id": sample_id, "passed": False, "error": "Failed to extract Python code",
            "generated_code": "", "model_response": model_response,
        }

    if function_signature:
        if not check_function_signature(generated_code, function_signature):
            return {
                "id": sample_id, "passed": False, "error": "Incorrect function signature",
                "generated_code": generated_code, "model_response": model_response,
                "expected_signature": function_signature,
            }

    if name:
        if not check_api_usage(generated_code, name, change_type, module):
            return {
                "id": sample_id, "passed": False,
                "error": f"API not used: {name}",
                "generated_code": generated_code, "model_response": model_response,
            }

    clean_code, clean_test = prepare_test_inputs(generated_code, test_program)

    actual_python = test_python
    sandbox_info = ""
    if use_sandbox and module and to_version:
        sandbox_python = get_sandbox_python(
            module.split('.')[0], to_version, sandbox_cache
        )
        if sandbox_python:
            actual_python = sandbox_python
            sandbox_info = f"sandbox:{module.split('.')[0]}@{to_version}"

    result = run_python_test(clean_code, clean_test, timeout=timeout,
                             python_bin=actual_python)

    test_output = result.get("output", "")
    tc_passed, tc_total = _parse_test_cases(test_output)
    return {
        "id": sample_id,
        "passed": result["success"],
        "error": result.get("error", ""),
        "test_output": test_output,
        "test_cases_passed": tc_passed,
        "test_cases_total": tc_total,
        "generated_code": generated_code,
        "model_response": model_response,
        "sandbox": sandbox_info,
    }


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PyEvo Baseline Evaluation (w/o API docs)")
    parser.add_argument("--data_path", default="data/PyEvo/PyEvo.json")
    parser.add_argument("--output", default="results/PyEvo/baseline_results.json")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--base_url", default="http://localhost:8003/v1")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TEST_TIMEOUT)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--start_id", type=int, default=None)
    parser.add_argument("--end_id", type=int, default=None)
    parser.add_argument("--test_python", default=DEFAULT_PYTHON_BIN,
                        help="Python interpreter for sandbox test execution (PyEvo env)")
    parser.add_argument("--sandbox", action="store_true",
                        help="Enable version-aware sandbox testing (each entry uses matching lib version)")
    parser.add_argument("--sandbox_cache",
                        default="data/sandbox_cache",
                        help="Directory to cache sandbox environments")
    args = parser.parse_args()

    print(f"Loading dataset: {args.data_path}")
    print(f"Test sandbox Python: {args.test_python}")
    if args.sandbox:
        print(f"Sandbox mode: ENABLED (cache: {args.sandbox_cache})")
    with open(args.data_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.start_id is not None or args.end_id is not None:
        s = args.start_id or 0
        e = args.end_id or len(data) - 1
        data = [d for i, d in enumerate(data) if s <= i <= e]
        print(f"Filtered to index {s}–{e}")

    print(f"Total: {len(data)} | Model: {args.model} | Workers: {args.max_workers}")

    # Resume support
    results = []
    processed_ids = set()
    if os.path.exists(args.output):
        try:
            with open(args.output, encoding="utf-8") as f:
                results = json.load(f)
            processed_ids = {r["id"] for r in results}
            print(f"Resuming: {len(results)} already done")
        except Exception:
            pass

    remaining = [(i, s) for i, s in enumerate(data)
                 if f"{i}_{s.get('name', 'unknown')}" not in processed_ids]
    print(f"Remaining: {len(remaining)}")
    if not remaining:
        print("All done!")
        _print_summary(results)
        return

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    with tqdm(total=len(remaining), desc="Baseline eval (w/o API)") as pbar:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_sample, s, i, args.model, args.base_url,
                    args.timeout, args.max_tokens, args.test_python,
                    args.sandbox, args.sandbox_cache,
                ): i
                for i, s in remaining
            }
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                    if r["passed"]:
                        passed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    failed += 1
                    print(f"\nException: {exc}")

                pbar.update(1)
                pbar.set_postfix(
                    passed=passed, failed=failed,
                    rate=f"{passed/(passed+failed)*100:.1f}%" if passed + failed else "0%",
                )

                if len(results) % 20 == 0:
                    _save(results, args.output)

    _save(results, args.output)
    _print_summary(results)


def _save(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _print_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    from collections import Counter
    errors = Counter(r.get("error", "") for r in results if not r["passed"])

    # AUA: exclude signature/api errors
    sig_err = sum(1 for r in results if "Incorrect function signature" in r.get("error", ""))
    api_err = sum(1 for r in results if r.get("error", "").startswith("API not used"))
    aua_n   = total - sig_err - api_err
    aua_pct = aua_n / total * 100 if total else 0.0

    # Coverage: test-case level
    tc_passed = sum(r.get("test_cases_passed", 0) for r in results)
    tc_total  = sum(r.get("test_cases_total",  0) for r in results)
    if tc_total == 0:  # backward compat: parse from test_output
        for r in results:
            p, t = _parse_test_cases(r.get("test_output", ""))
            tc_passed += p; tc_total += t
    coverage_pct = tc_passed / tc_total * 100 if tc_total else 0.0

    print(f"\n{'='*60}")
    print(f"PyEvo Baseline (w/o API) Results")
    print(f"{'='*60}")
    print(f"Total: {total}")

    for k in [1, 2, 5]:
        if k <= total:
            print(f"  pass@{k}: {pass_at_k(total, passed, k)*100:.2f}%")

    if total:
        print(f"\nPassed:   {passed}/{total} ({passed/total*100:.2f}%)")
        print(f"AUA:      {aua_n}/{total} ({aua_pct:.2f}%)")
        print(f"Coverage: {tc_passed}/{tc_total} test cases ({coverage_pct:.2f}%)")
    if errors:
        print(f"\nError breakdown:")
        for err, cnt in errors.most_common(10):
            print(f"  {err:30s}: {cnt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
