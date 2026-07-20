"""Validate RustEvo samples with the public Rust test harness."""
import json
import sys
import os
import re
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from util.rust_cargo_test import (
    run_rust_test_auto,
    check_function_signature,
    check_api_usage,
    prepare_rust_test_inputs,
    DEFAULT_RUST_VERSION,
    DEFAULT_TEST_TIMEOUT
)



#=
#=

def validate_sample(sample: dict, index: int, timeout: int = DEFAULT_TEST_TIMEOUT) -> dict:
    """
    Validate a single sample
    
    Core logic:
    1. Basic check (code and test exist)
    2. API usage check (deprecated samples must not use old API)
    3. Function signature check (if signature field exists)
    4. Detect if third-party crates are used in code
    5. Select test strategy based on third-party vs stdlib:
       - third-party library: cargo test, Rust version pinned to 1.84.0, library version read from to_version
       - standard library: rustc --test, Rust version read from to_version, mapped to installed range
    """
    code = sample.get('code', '')
    test_program = sample.get('test_program', '')
    sample_name = sample.get('name', 'unknown')
    change_type = sample.get('change_type', '')
    to_version = sample.get('to_version', '')
    signature = sample.get('signature', '')
    function_signature = sample.get('function_signature', '')
    
    if not test_program:
        return {
            'index': index, 'name': sample_name, 'passed': False,
            'error': 'No test program', 'error_type': 'missing_test',
            'version': DEFAULT_RUST_VERSION, 'third_party': False, 'crates': []
        }
    
    if not code:
        return {
            'index': index, 'name': sample_name, 'passed': False,
            'error': 'No code', 'error_type': 'missing_code',
            'version': DEFAULT_RUST_VERSION, 'third_party': False, 'crates': []
        }
    
    if sample_name:
        api_module = sample.get('module', '')
        replacement_api = sample.get('replacement_api', '')
        if not check_api_usage(code, sample_name, change_type, api_module, test_program, replacement_api):
            error_msg = (
                f'Deprecated API used: {sample_name}'
                if str(change_type).lower() == 'deprecated'
                else f'Required API not used: {sample_name}'
            )
            return {
                'index': index, 'name': sample_name, 'passed': False,
                'error': error_msg, 'error_type': 'api_usage',
                'version': DEFAULT_RUST_VERSION, 'third_party': False, 'crates': []
            }
    
    if function_signature and not check_function_signature(code, function_signature):
        return {
            'index': index, 'name': sample_name, 'passed': False,
            'error': f'Required function signature not found: {function_signature}',
            'error_type': 'missing_function_signature',
            'version': DEFAULT_RUST_VERSION, 'third_party': False, 'crates': []
        }

    module = sample.get('module', '')
    clean_code, clean_test, rust_version, crate_version, is_third_party, crate_names = prepare_rust_test_inputs(
        code,
        test_program,
        module,
        to_version,
        dedup_imports=True,
    )
    
    result = run_rust_test_auto(
        code=clean_code,
        test_code=clean_test,
        module=module,
        rust_version=rust_version,
        crate_version=crate_version,
        timeout=timeout
    )
    
    return {
        'index': index,
        'name': sample_name,
        'passed': result['success'],
        'error': result.get('error', ''),
        'error_type': result.get('error_type', 'unknown'),
        'status': result.get('status', 'UNKNOWN'),
        'version': rust_version,
        'third_party': is_third_party,
        'crates': list(crate_names) if is_third_party else [],
        'stdout': result.get('stdout', '')[:500],
        'stderr': result.get('stderr', '')[:500],
    }


#=
#=

def main():
    parser = argparse.ArgumentParser(description='Validate RustEvo dataset using rustc/cargo')
    parser.add_argument('--data_path', type=str, default=str(Path(__file__).resolve().parent.parent / 'data' / 'RustEvo' / 'RustEvo.json'),
                        help='dataset path')
    parser.add_argument('--output', type=str, default=str(Path(__file__).resolve().parent.parent / 'results' / 'rustevo_validation.json'),
                        help='output results path')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TEST_TIMEOUT,
                        help='Timeout per sample (seconds)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of test samples')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting sample index')
    
    args = parser.parse_args()
    
    #  python validate_rustevo.py --limit 110 --start 620

    print(f"Load dataset: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.limit:
        data = data[args.start:args.start + args.limit]
    elif args.start > 0:
        data = data[args.start:]
    
    print(f"Samples to validate: {len(data)}")
    
    results = []
    passed_count = 0
    failed_samples = []
    third_party_count = 0
    third_party_passed = 0
    version_stats = defaultdict(lambda: {'passed': 0, 'failed': 0})
    crate_usage = Counter()
    error_type_stats = Counter()
    
    for i, sample in enumerate(tqdm(data, desc="Validating samples"), args.start):
        result = validate_sample(sample, i, timeout=args.timeout)
        results.append(result)
        
        if result.get('third_party', False):
            third_party_count += 1
            if result['passed']:
                third_party_passed += 1
            for crate in result.get('crates', []):
                crate_usage[crate] += 1
        
        if result['passed']:
            passed_count += 1
            version_stats[result['version']]['passed'] += 1
        else:
            failed_samples.append(result)
            version_stats[result['version']]['failed'] += 1
            error_type = result.get('error_type', 'unknown')
            error_type_stats[error_type] += 1
    
    print("\n" + "="*70)
    print("Validation results")
    print("="*70)
    print(f"Total samples: {len(data)}")
    print(f"[PASS] passed: {passed_count}")
    print(f"[FAIL] failed: {len(failed_samples)}")
    print(f"\nPass rate: {passed_count / len(data) * 100:.1f}%")
    
    print(f"\nThird-party library samples: {third_party_count}")
    print(f" [PASS] passed: {third_party_passed}")
    print(f" [FAIL] failed: {third_party_count - third_party_passed}")
    if third_party_count > 0:
        print(f" Pass rate: {third_party_passed / third_party_count * 100:.1f}%")
    
    stdlib_count = len(data) - third_party_count
    stdlib_passed = passed_count - third_party_passed
    print(f"\nPure stdlib samples: {stdlib_count}")
    print(f" [PASS] passed: {stdlib_passed}")
    print(f" [FAIL] failed: {stdlib_count - stdlib_passed}")
    if stdlib_count > 0:
        print(f" Pass rate: {stdlib_passed / stdlib_count * 100:.1f}%")
    
    if crate_usage:
        print("\n" + "="*70)
        print("third-party libraryUse:")
        print("="*70)
        for crate, count in crate_usage.most_common():
            print(f" {crate}: {count} sample")
    
    if error_type_stats:
        print("\n" + "="*70)
        print("errortype:")
        print("="*70)
        for error_type, count in error_type_stats.most_common():
            print(f" {error_type}: {count} sample")
    
    print("\n" + "="*70)
    print("Rustversiontest (Top 10):")
    print("="*70)
    sorted_versions = sorted(version_stats.items(), 
                            key=lambda x: x[1]['passed'] + x[1]['failed'], 
                            reverse=True)[:10]
    for version, stats in sorted_versions:
        total = stats['passed'] + stats['failed']
        rate = stats['passed'] / total * 100 if total > 0 else 0
        print(f" v{version}: {stats['passed']}/{total} passed ({rate:.1f}%)")
    
    output_data = {
        'summary': {
            'total': len(data),
            'passed': passed_count,
            'failed': len(failed_samples),
            'pass_rate': passed_count / len(data) if len(data) > 0 else 0,
            'third_party_total': third_party_count,
            'third_party_passed': third_party_passed,
            'stdlib_total': stdlib_count,
            'stdlib_passed': stdlib_passed,
        },
        'version_stats': dict(version_stats),
        'crate_usage': dict(crate_usage),
        'error_type_stats': dict(error_type_stats),
        'failed_samples': failed_samples,
        'all_results': results,
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailedResults saved to: {args.output}")
    
    if failed_samples:
        print(f"\nfailedsample ({len(failed_samples)}):")
        for s in failed_samples[:20]:
            third_party_mark = " [third-party library]" if s.get('third_party', False) else ""
            error_type = s.get('error_type', 'unknown')
            status = s.get('status', 'UNKNOWN')
            print(f"  #{s['index']}: {s['name']} (v{s['version']}) [{error_type}] {status}{third_party_mark}")
            error_preview = s.get('error', '')[:100] if s.get('error') else 'No error message'
            print(f"      {error_preview}...")
        if len(failed_samples) > 20:
            print(f" ... {len(failed_samples) - 20} failedsample")


if __name__ == '__main__':
    main()
