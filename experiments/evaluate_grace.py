"""
Evaluate GRACE and A-GRACE on RustEvo Dataset

Following evaluate_gnn.py structure, evaluate GRACE A-GRACE method

Usage:
    python experiments/evaluate_grace.py \
        --alg_name AGRACE \
        --model_name /path/to/model \
        --hparams_fname Qwen2.5-7B-Instruct.json \
        --dataset_size_limit 1000
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import random
import re
import argparse

from dsets import RustEvoDataset

from methods.Baselines.GRACE import GraceHyperParams, apply_grace_to_model, restore_grace_model
from methods.Baselines.AGRACE import AGraceHyperParams, apply_agrace_to_model, restore_agrace_model

from util import nethook
from util.globals import *
from util.rust_cargo_test import (
    run_rust_test_auto,
    check_function_signature,
    check_api_usage,
    prepare_rust_test_inputs,
    DEFAULT_TEST_TIMEOUT
)

# Algorithm registry
ALG_DICT = {
    "GRACE": (GraceHyperParams, apply_grace_to_model),
    "AGRACE": (AGraceHyperParams, apply_agrace_to_model),
}


def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""


def extract_rust_code(text: str) -> str:
    """Extract Rust code block from LLM output."""
    matches = re.findall(r"```(?:rust)?\s*(.*?)```", text, re.DOTALL)
    if matches:
        valid_matches = [m.strip() for m in matches if m.strip()]
        if valid_matches:
            return max(valid_matches, key=len)
    return text.strip("` \n\t")


def get_code_generation_prompt(sample: dict) -> str:
    """Buildcodegenerateprompt"""
    query = sample.get("query", "")
    function_signature = sample.get("function_signature", "")
    name = sample.get("name", "")
    module = sample.get("module", "")

    is_crate = not (module.startswith("std::") or module.startswith("core::") or module.startswith("alloc::"))

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

    Requirements:
    1. Implement ONLY the Rust function with the signature given above.
    2. Your implementation MUST use the specified API: {name}
    3. Do not include tests or any extra comments.

    Respond with ONLY the Rust function implementation.
"""
    else:
        crate_name = module.split("::")[0] if "::" in module else module

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

    Requirements:
    1. Implement ONLY the Rust function with the signature given above.
    2. Your implementation MUST use the specified API: {name}
    3. Compile with Rust 1.84.0. Do not include tests or any extra comments.

    Respond with ONLY the Rust function implementation.
"""    
    return prompt


def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int = 1,
):
    set_seed()
    
    if alg_name not in ALG_DICT:
        raise ValueError(f"Unknown algorithm: {alg_name}. Available: {list(ALG_DICT.keys())}")
    
    params_class, apply_algo = ALG_DICT[alg_name]
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    
    print(f"\n{'='*80}")
    print(f"RustEvo {alg_name} Evaluation")
    print(f"{'='*80}")
    print(f"Algorithm: {alg_name}")
    print(f"Model: {model_name}")
    print(f"Hparams: {hparams_fname}")
    print(f"{'='*80}\n")

    # Load model
    if isinstance(model_name, str):
        print("Loading model...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            attn_implementation="eager"
        ).cuda()
        
        tok = AutoTokenizer.from_pretrained(model_name)
        # Set pad token
        if tok.pad_token is None:
            if tok.unk_token is not None:
                tok.pad_token = tok.unk_token
            else:
                tok.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tok))
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load dataset
    ds = RustEvoDataset(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)
    print(f"Dataset loaded: {len(ds)} samples")
    
    # Load alpaca data for stability
    with open(Path(DATA_DIR) / "alpaca_data.json", 'r', encoding='utf-8') as f:
        ex_datas = json.load(f)
    
    if hparams.model_name in ['Llama3-8B-Instruct', 'Llama3.1-8B-Instruct']:
        ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output'] for i in ex_datas]
    elif hparams.model_name == 'Qwen2.5-7B-Instruct':
        ex_datas = [get_qwen_without_answer(i['instruction']+i['input'])+i['output'] for i in ex_datas]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    # Set pad token
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    
    edited_data = []
    
    print(f"\n{'='*80}")
    print(f"RustEvo: Edit -> Generate -> Test (per sample) using {alg_name}")
    print(f"{'='*80}\n")
    
    overall_start = time()
    test_results = {
        'total': len(ds),
        'passed': 0,
        'no_test': 0,
        'compilation_failed': 0,
        'test_failed': 0,
        'timeout': 0,
        'signature_error': 0,
        'api_error': 0,
        'extraction_error': 0,
        'other_error': 0,
        'failed_samples': []
    }
    
    for sample_idx in tqdm(range(len(ds)), desc=f"Processing RustEvo with {alg_name}"):
        data = ds[sample_idx]
        batch = [data]
        sample_id = data.get('id', f'sample_{sample_idx}')
        
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}/{len(ds)}, ID: {sample_id}")
        print(f"{'='*60}")
        
        # Step 1: Apply edit
        edit_start = time()
        
        weights_copy = apply_algo(model, tok, hparams, batch)
        
        edit_time = time() - edit_start
        print(f"[OK] Edit complete: {edit_time:.2f}s")
        
        # Step 2: Generate code
        gen_start = time()
        test_prompt = get_code_generation_prompt(data)
        
        # Format prompt with chat template
        if 'Llama' in model_name or 'llama' in model_name:
            formatted_prompt = get_llama_without_answer(test_prompt)
        elif 'Qwen' in model_name or 'qwen' in model_name:
            formatted_prompt = get_qwen_without_answer(test_prompt)
        else:
            formatted_prompt = test_prompt
        
        question = tokenizer([formatted_prompt], return_tensors='pt', padding=True)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=question['input_ids'].to('cuda'),
                attention_mask=question['attention_mask'].to('cuda'),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=model.config.eos_token_id,
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(question['input_ids'], generated_ids)
        ]
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        data['original_prediction'] = output[0]
        gen_time = time() - gen_start
        print(f"[OK] Generation complete: {gen_time:.2f}s")
        
        print(f"\n{'─'*60}")
        print(f"Question: {data['question'][:30]}...")
        print(f"\nComplete LLM Output:")
        print(f"{'-'*60}")
        print(output[0])
        print(f"{'-'*60}")
        
        test_start = time()
        generated_code = extract_rust_code(data.get('original_prediction', ''))
        print(f"\n📦 Extracted Code: {generated_code[:80]}..." if len(generated_code) > 80 else f"\n📦 Extracted Code: {generated_code}")
        
        if not generated_code or len(generated_code.strip()) < 10:
            data['test_status'] = 'EXTRACTION_FAILED'
            data['test_error'] = 'Failed to extract valid code'
            data['extracted_code'] = generated_code
            test_results['extraction_error'] += 1
            print(f"[FAIL] Failed: Code extraction failed")
            edited_data.append(data)
            continue
        
        test_program = data.get('test_program', '')
        
        if not test_program or test_program == 'INCORRECT CODE':
            data['test_status'] = 'NO_TEST'
            data['test_error'] = 'No valid test program'
            data['extracted_code'] = generated_code
            test_results['no_test'] += 1
            print(f"[Warn]  No test program")
        else:
            function_signature = data.get('function_signature', '')
            if function_signature:
                if not check_function_signature(generated_code, function_signature):
                    data['test_status'] = 'SIGNATURE_ERROR'
                    data['test_passed'] = False
                    data['test_error'] = f'Incorrect function signature: {function_signature}'
                    data['extracted_code'] = generated_code
                    test_results['signature_error'] += 1
                    test_results['failed_samples'].append({
                        'id': sample_id, 'status': 'FAILED', 'error': 'Incorrect function signature'
                    })
                    print(f"[FAIL] Failed: Incorrect function signature")
                    
                    edited_data.append(data)
                    continue
            
            api_name = data.get('name', '')
            change_type = data.get('change_type', '')
            api_module = data.get('module', '')
            if api_name:
                replacement_api = data.get('replacement_api', '')
                if not check_api_usage(generated_code, api_name, change_type, api_module, test_program, replacement_api):
                    error_msg = (
                        f'Deprecated API used: {api_name}'
                        if str(change_type).lower() == 'deprecated'
                        else f'Required API not used: {api_name}'
                    )
                    data['test_status'] = 'API_ERROR'
                    data['test_passed'] = False
                    data['test_error'] = error_msg
                    data['extracted_code'] = generated_code
                    test_results['api_error'] += 1
                    test_results['failed_samples'].append({
                        'id': sample_id, 'status': 'FAILED', 'error': error_msg
                    })
                    print(f"[FAIL] Failed: {error_msg}")
                    
                    edited_data.append(data)
                    continue
            
            api_module = data.get('module', '')
            to_version = data.get('to_version', '1.84.0')
            
            clean_code, clean_test, rust_version, crate_version, _, _ = prepare_rust_test_inputs(
                generated_code,
                test_program,
                api_module,
                to_version,
                dedup_imports=True,
            )
            
            result = run_rust_test_auto(
                clean_code,
                clean_test,
                api_module,
                rust_version,
                crate_version,
                timeout=DEFAULT_TEST_TIMEOUT
            )
            
            data['test_status'] = result['status']
            data['test_passed'] = result['success']
            data['extracted_code'] = generated_code
            
            if result.get('error'):
                data['test_error'] = result['error']
            if result.get('stderr'):
                data['test_stderr'] = result['stderr']
            
            if result['success']:
                test_results['passed'] += 1
                print(f"[PASS] Test passed")
            else:
                error_type = result.get('error_type', 'other')
                if error_type == 'compilation':
                    test_results['compilation_failed'] += 1
                elif error_type == 'test_failed':
                    test_results['test_failed'] += 1
                elif error_type == 'timeout':
                    test_results['timeout'] += 1
                else:
                    test_results['other_error'] += 1
                
                test_results['failed_samples'].append({
                    'id': sample_id, 'status': result['status'], 'error': result.get('error', '')[:200]
                })
                print(f"[FAIL] Test failed: {result['status']}")
            
            test_time = time() - test_start
            print(f"[OK] Test complete: {test_time:.2f}s")
        
        if alg_name == "GRACE":
            restore_grace_model(model, weights_copy)
        elif alg_name == "AGRACE":
            restore_agrace_model(model, weights_copy)
        
        edited_data.append(data)
        
        if (sample_idx + 1) % 10 == 0 or sample_idx == len(ds) - 1:
            tested = sample_idx + 1 - test_results['no_test']
            failed = tested - test_results['passed']
            pass_rate = test_results['passed'] / tested * 100 if tested > 0 else 0
            fail_rate = failed / tested * 100 if tested > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"[Stats] Progress: {sample_idx + 1}/{len(ds)} samples processed")
            print(f"[PASS] Passed: {test_results['passed']}/{tested} ({pass_rate:.1f}%)")
            print(f"[FAIL] Failed: {failed}/{tested} ({fail_rate:.1f}%)")
            if test_results['compilation_failed'] > 0:
                print(f"   - Compilation: {test_results['compilation_failed']}")
            if test_results['test_failed'] > 0:
                print(f"   - Test failed: {test_results['test_failed']}")
            if test_results['timeout'] > 0:
                print(f"   - Timeout: {test_results['timeout']}")
            if test_results['other_error'] > 0:
                print(f"   - Other (signature/API): {test_results['other_error']}")
            if test_results['no_test'] > 0:
                print(f"[Warn]  No test: {test_results['no_test']}")
            print(f"{'='*60}")
    
    tested_count = test_results['total'] - test_results['no_test']
    failed_count = test_results['total'] - test_results['passed'] - test_results['no_test']
    
    print(f"\n{'='*80}")
    print(f"RustEvo Test Results - {alg_name}")
    print(f"{'='*80}")
    print(f"Total samples: {test_results['total']}")
    print(f"Tested samples: {tested_count}")
    
    print(f"\n[PASS] Passed: {test_results['passed']}/{tested_count} ({test_results['passed']/tested_count*100:.2f}%)" if tested_count > 0 else "\n[PASS] Passed: 0")
    
    print(f"\n[FAIL] Failed: {failed_count}/{tested_count} ({failed_count/tested_count*100:.2f}%)" if tested_count > 0 else "\n[FAIL] Failed: 0")
    if test_results['compilation_failed'] > 0:
        comp_rate = test_results['compilation_failed']/tested_count*100 if tested_count > 0 else 0
        print(f"   - Compilation errors: {test_results['compilation_failed']} ({comp_rate:.1f}%)")
    if test_results['test_failed'] > 0:
        test_rate = test_results['test_failed']/tested_count*100 if tested_count > 0 else 0
        print(f"   - Test execution failed: {test_results['test_failed']} ({test_rate:.1f}%)")
    if test_results['timeout'] > 0:
        timeout_rate = test_results['timeout']/tested_count*100 if tested_count > 0 else 0
        print(f"   - Timeout: {test_results['timeout']} ({timeout_rate:.1f}%)")
    if test_results['signature_error'] > 0:
        sig_rate = test_results['signature_error']/tested_count*100 if tested_count > 0 else 0
        print(f" - Function signature error: {test_results['signature_error']} ({sig_rate:.1f}%)")
    if test_results['api_error'] > 0:
        api_rate = test_results['api_error']/tested_count*100 if tested_count > 0 else 0
        print(f" - APIUseerror: {test_results['api_error']} ({api_rate:.1f}%)")
    if test_results['extraction_error'] > 0:
        ext_rate = test_results['extraction_error']/tested_count*100 if tested_count > 0 else 0
        print(f" - Code extraction failed: {test_results['extraction_error']} ({ext_rate:.1f}%)")
    if test_results['other_error'] > 0:
        other_rate = test_results['other_error']/tested_count*100 if tested_count > 0 else 0
        print(f" - Other error: {test_results['other_error']} ({other_rate:.1f}%)")
    
    if test_results['no_test'] > 0:
        no_test_rate = test_results['no_test']/test_results['total']*100
        print(f"\n[Warn]  No test program: {test_results['no_test']}/{test_results['total']} ({no_test_rate:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"Total time: {time() - overall_start:.2f}s")
    print(f"Average time per sample: {(time() - overall_start)/test_results['total']:.2f}s")
    print(f"{'='*80}")
    
    os.makedirs('output', exist_ok=True)
    output_file = f'output/{alg_name}_{hparams.model_name}_rustevo_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(edited_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["GRACE", "AGRACE"],
        default="GRACE",
        help="Editing algorithm to use.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="/path/to/model",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Qwen2.5-7B-Instruct.json",
        help="Name of hyperparameters file.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        default="RustEvo",
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate dataset to first n samples.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of edits per sample.",
    )

    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.num_edits,
    )
