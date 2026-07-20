"""
Evaluate model editing methods on RustEvo / PyEvo benchmarks.

Usage:
    # 1. Build precomputed graphs
    python scripts/build_rustevo_graphs.py --output_dir ./data/rustevo_graphs
    python scripts/build_pyevo_graphs.py  --output_dir ./data/pyevo_graphs

    # 2. Run evaluation
    python experiments/evaluate_gnn.py \\
        --alg_name MEMIT_ARE_AEG \\
        --model_name /path/to/model \\
        --hparams_fname Qwen2.5-7B-Instruct.json \\
        --ds_name rustevo \\
        --graph_dir ./data/rustevo_graphs \\
        --dataset_size_limit 200

Supported algorithms: MEMIT_ARE_AEG, AlphaEdit_ARE_AEG, UnKE_ARE_AEG,
                      MEMIT_ARE, AlphaEdit_ARE, UnKE_ARE, ROME, GRACE, AGRACE,
                      STAR
Supported datasets:   rustevo, pyevo
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pickle
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import random
import re

from dsets import RustEvoDataset, PyEvoDataset

from methods.Baselines.AlphaEdit_ARE import AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model, get_cov

from methods.AEG_Edit.AlphaEdit_ARE_AEG import AlphaEditAREAEGHyperParams, apply_alphaedit_are_aeg_to_model
from methods.Baselines.UnKE_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from methods.AEG_Edit.UnKE_ARE_AEG import unkeAREAEGHyperParams, apply_unke_are_aeg_to_model
from methods.Baselines.MEMIT_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from methods.AEG_Edit.MEMIT_ARE_AEG import MEMITAREAEGHyperParams, apply_memit_are_aeg_to_model
from methods.Baselines.ROME import ROMEHyperParams, apply_rome_to_model
from methods.Baselines.GRACE import GraceHyperParams, apply_grace_to_model, restore_grace_model
from methods.Baselines.AGRACE import AGraceHyperParams, apply_agrace_to_model, restore_agrace_model
from methods.Baselines.STAR import STARHyperParams, apply_STAR_to_model


from util import nethook
from util.globals import *
from util.rust_cargo_test import (
    run_rust_test_auto,
    check_function_signature,
    check_api_usage,
    prepare_rust_test_inputs,
    DEFAULT_TEST_TIMEOUT
)
from util.python_pytest_test import (
    run_python_test,
    check_function_signature as py_check_signature,
    check_api_usage as py_check_api_usage,
    get_sandbox_python,
    DEFAULT_PYTHON_BIN,
    DEFAULT_TEST_TIMEOUT as PY_DEFAULT_TIMEOUT,
)


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


# Algorithm registry
ALG_DICT = {
    "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    "AlphaEdit_ARE_AEG": (AlphaEditAREAEGHyperParams, apply_alphaedit_are_aeg_to_model),
    "UnKE_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "UnKE_ARE_AEG": (unkeAREAEGHyperParams, apply_unke_are_aeg_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT_ARE_AEG": (MEMITAREAEGHyperParams, apply_memit_are_aeg_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "GRACE": (GraceHyperParams, apply_grace_to_model),
    "AGRACE": (AGraceHyperParams, apply_agrace_to_model),
    "STAR": (STARHyperParams, apply_STAR_to_model),
}


def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""


def extract_rust_code(text: str) -> str:
    """Extract Rust code block from LLM output."""
    # Strategy 1: fenced code block with language tag
    lang_blocks = re.findall(r"```(?:rust|rs|Rust)\b[^\n]*\n([\s\S]*?)```", text)
    if lang_blocks:
        valid = [m.strip() for m in lang_blocks if m.strip()]
        if valid:
            return max(valid, key=len)
    # Strategy 2: bare fenced code block
    bare_blocks = re.findall(r"```\s*\n([\s\S]*?)```", text)
    if bare_blocks:
        valid = [m.strip() for m in bare_blocks if m.strip()]
        if valid:
            return max(valid, key=len)
    # Strategy 3: trailing fence only
    candidate = re.sub(r"```\s*$", "", text.strip())
    if candidate and '```' not in candidate:
        return candidate.strip()
    return text.strip("` \n\t")


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM output."""
    code = None

    lang_blocks = re.findall(r"```(?:[Pp]ython[23]?|py)\b[^\n]*\n([\s\S]*?)```", text)
    if lang_blocks:
        code = max(lang_blocks, key=len).strip()

    if code is None:
        bare_blocks = re.findall(r"```\s*\n([\s\S]*?)```", text)
        if bare_blocks:
            code_blocks = [m.strip() for m in bare_blocks
                           if m.strip() and ('def ' in m or 'import ' in m)]
            if code_blocks:
                code = max(code_blocks, key=len)
            else:
                valid = [m.strip() for m in bare_blocks if m.strip()]
                if valid:
                    code = max(valid, key=len)

    if code is None and '[PYTHON]' in text and '[/PYTHON]' in text:
        start = text.find('[PYTHON]') + len('[PYTHON]')
        end = text.find('[/PYTHON]')
        if end > start:
            code = text[start:end].strip()

    if code is None:
        candidate = re.sub(r"```\s*$", "", text.strip())
        if candidate and '```' not in candidate:
            code = candidate.strip()

    if code is None:
        return text.strip()

    # Filter out test functions and test calls
    lines = code.split('\n')
    result_lines = []
    in_test_func = False
    main_func_indent = None

    for line in lines:
        stripped = line.strip()

        # Skip blank lines (preserve within functions)
        if not stripped:
            if result_lines and not in_test_func:
                result_lines.append(line)
            continue

        # Detect test function definitions
        if stripped.startswith('def test_') or stripped.startswith('async def test_'):
            in_test_func = True
            continue

        # Inside test function body
        if in_test_func:
            if line.startswith((' ', '\t')):
                continue
            else:
                in_test_func = False

        # Skip test calls
        if stripped.startswith('test_') and '(' in stripped:
            continue

        if stripped.startswith('#') and 'test' in stripped.lower():
            continue

        # Collect imports
        if stripped.startswith(('import ', 'from ')):
            result_lines.append(line)
            continue

        # Collect main function definition
        is_func_def = stripped.startswith('def ') or stripped.startswith('async def ')
        is_test_def = stripped.startswith('def test_') or stripped.startswith('async def test_')
        if is_func_def and not is_test_def:
            main_func_indent = len(line) - len(line.lstrip())
            result_lines.append(line)
            continue

        # Collect function body
        if main_func_indent is not None:
            current_indent = len(line) - len(line.lstrip()) if stripped else 0
            if current_indent > main_func_indent or not stripped:
                result_lines.append(line)
            elif (stripped.startswith('def ') or stripped.startswith('async def ')) and not (stripped.startswith('def test_') or stripped.startswith('async def test_')):
                main_func_indent = len(line) - len(line.lstrip())
                result_lines.append(line)
            elif current_indent == main_func_indent and not (stripped.startswith('def test_') or stripped.startswith('async def test_')) and not (stripped.startswith('test_') and '(' in stripped):
                result_lines.append(line)

    # Trim trailing blank lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    return '\n'.join(result_lines) if result_lines else text.strip()


def get_rustevo_prompt(sample: dict) -> str:
    """Build code generation prompt for RustEvo."""
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




def get_pyevo_prompt(sample: dict) -> str:
    """Build code generation prompt for PyEvo."""
    name               = sample.get('name', '')
    module             = sample.get('module', '')
    query              = sample.get('query', '')
    # signature          = sample.get('signature', '')
    # documentation      = sample.get('documentation', '')
    # source_code        = sample.get('source_code', '')
    # from_version       = sample.get('from_version', '')
    # to_version         = sample.get('to_version', '')
    function_signature = sample.get('function_signature', '').strip()

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


def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_project(model, tok, layer, hparams):
    """Compute null-space projection matrix."""
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    )
    P = torch.eye(cov.shape[0]).cuda() - cov.cuda() @ torch.linalg.solve(
        hparams.nullspace_threshold * torch.eye(cov.shape[0]).cuda() + cov.cuda(),
        cov.cuda()
    )
    return P


def load_prebuilt_graphs(graph_dir: str, ds_name: str) -> list:
    """Load prebuilt graphs from disk."""
    # Select graph file by dataset
    if ds_name == "pyevo":
        graph_file = Path(graph_dir) / "pyevo_graphs.json"
        build_script = "scripts/build_pyevo_graphs.py"
    else:
        graph_file = Path(graph_dir) / "rustevo_graphs.json"
        build_script = "scripts/build_rustevo_graphs.py"

    if not graph_file.exists():
        print(f"Warning: Graph file not found: {graph_file}")
        print(f"Please run {build_script} first")
        return None

    print(f"Loading prebuilt graphs from {graph_file}...")
    with open(graph_file, 'r', encoding='utf-8') as f:
        graphs = json.load(f)
 
    valid_count = sum(1 for g in graphs if g is not None)
    print(f"Loaded {len(graphs)} graphs ({valid_count} valid)")

    return graphs


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    graph_dir: str,
    num_edits: int = 1,
):
    set_seed()
    
    if alg_name not in ALG_DICT:
        raise ValueError(f"Unknown algorithm: {alg_name}. Available: {list(ALG_DICT.keys())}")
    
    params_class, apply_algo = ALG_DICT[alg_name]
    params_path = HPARAMS_DIR / alg_name / hparams_fname
    hparams = params_class.from_json(params_path)
    
    print(f"\n{'='*80}")
    print(f"Code Evolution AEG Evaluation")
    print(f"{'='*80}")
    print(f"Algorithm: {alg_name}")
    print(f"Model: {model_name}")
    print(f"Dataset: {ds_name}")
    print(f"Hparams: {hparams_fname}")
    print(f"Graph dir: {graph_dir}")
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
    # Base data directory
    DATA_DIR = "data"
    
    if ds_name == "rustevo":
        ds = RustEvoDataset(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)
        dataset_type = "rustevo"
    elif ds_name == "pyevo":
        ds = PyEvoDataset(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)
        dataset_type = "pyevo"
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")
    
    print(f"Dataset '{ds_name}' loaded: {len(ds)} samples")
    print(f"Model name for formatting: {hparams.model_name}")
    
    # Load alpaca data for stability
    with open(Path(DATA_DIR) / "alpaca_data.json", 'r', encoding='utf-8') as f:
        ex_datas = json.load(f)
    
    # Format alpaca data for model
    if any(name in hparams.model_name for name in ['Llama3', 'Llama-3', 'llama3', 'llama-3']):
        ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output'] for i in ex_datas]
    elif 'Qwen' in hparams.model_name or 'qwen' in hparams.model_name:
        ex_datas = [get_qwen_without_answer(i['instruction']+i['input'])+i['output'] for i in ex_datas]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    # Set pad token
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    # Load prebuilt graphs
    prebuilt_graphs = None
    aeg_methods = ["AlphaEdit_ARE_AEG", "UnKE_ARE_AEG", "MEMIT_ARE_AEG"]
    if alg_name in aeg_methods and graph_dir:
        prebuilt_graphs = load_prebuilt_graphs(graph_dir, ds_name)
        if prebuilt_graphs is None:
            print("Warning: No prebuilt graphs, AEG enhancement will be disabled")

    # Compute null-space projection (AlphaEdit only)
    P = None
    if any(alg in alg_name for alg in ["AlphaEdit", "AlphaEdit_ARE"]):
        proj_file = f"{hparams.model_name}_null_space_project.pt"
        if not os.path.exists(proj_file):
            print("Computing null space projection matrix...")
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            del W_out
            for i, layer in enumerate(hparams.layers):
                P[i, :, :] = get_project(model, tok, layer, hparams)
            torch.save(P, proj_file)
        else:
            P = torch.load(proj_file)
    
    edited_data = []
    
    # Evaluation loop
    print(f"\n{'='*80}")
    if dataset_type == "rustevo":
        print("RustEvo: Edit -> Generate -> Test (per sample)")
    elif dataset_type == "pyevo":
        print("PyEvo: Edit -> Generate -> Test (per sample)")

    print(f"{'='*80}\n")

    overall_start = time()
    test_results = {
        'total': len(ds),
        'passed': 0,              # pass@k(new): all_pass_w_update = True
        'no_test': 0,
        'compilation_failed': 0,  # RustEvo only
        'test_failed': 0,         # Both
        'timeout': 0,             # Both
        'signature_error': 0,     # RustEvo only
        'api_error': 0,           # RustEvo only
        'extraction_error': 0,    # Both
        'other_error': 0,         # Both
        'test_cases_passed': 0,   # RustEvo/PyEvo: individual test case passes (for Coverage)
        'test_cases_total': 0,    # RustEvo/PyEvo: individual test case total (for Coverage)
        'failed_samples': []
    }
    
    # Progress snapshots
    progress_snapshots = []

    desc = f"Processing {ds_name}"
    for sample_idx in tqdm(range(len(ds)), desc=desc):
        data = ds[sample_idx]
        batch = [data]
        sample_id = data.get('id', f'sample_{sample_idx}')
        
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}/{len(ds)}, ID: {sample_id}")
        print(f"{'='*60}")
        
        try:
            # Get prebuilt graph for this sample
            graph_data_list = None
            if prebuilt_graphs is not None and sample_idx < len(prebuilt_graphs):
                graph_data = prebuilt_graphs[sample_idx]
                if graph_data is not None:
                    graph_data_list = [graph_data]
                    # Graph loaded
                    num_nodes = len(graph_data.get('nodes', []))
                    num_edges = len(graph_data.get('edges', []))
                    api_name = graph_data.get('api', {}).get('name', 'unknown')
                    print(f"Using prebuilt graph: API={api_name}, nodes={num_nodes}, edges={num_edges}")
                else:
                    print("No graph for this sample")

            # Step 1: Apply edit
            # Use short question for editing, full prompt for testing
            
            random_elements = random.sample(ex_datas, 20)
            nc_args = dict(P=P) if any(alg in alg_name for alg in ["AlphaEdit", "AlphaEdit_ARE"]) else dict()

            edit_start = time()

            if alg_name in aeg_methods:
                # AEG: pass prebuilt graph and dataset type
                weights_copy = apply_algo(
                    model, tok, hparams, batch,
                    ex_data=random_elements,  # Stability optimization data
                    **nc_args,
                    graph_data_list=graph_data_list,
                    dataset_type=dataset_type,
                )

            else:
                if alg_name == "UnKE_ARE":
                    weights_copy = apply_algo(model, tok, hparams, batch, random_elements, **nc_args)
                else:
                    weights_copy = apply_algo(model, tok, hparams, batch, **nc_args)
            
            edit_time = time() - edit_start
            print(f"[OK] Edit complete: {edit_time:.2f}s")
            
            # Step 2: Generate code
            gen_start = time()
            # Build prompt based on dataset type
            if dataset_type == "rustevo":
                test_prompt = get_rustevo_prompt(data)
            elif dataset_type == "pyevo":
                test_prompt = get_pyevo_prompt(data)
            else:
                test_prompt = get_pyevo_prompt(data)
        
            # Format prompt with chat template
            if any(name in hparams.model_name for name in ['Llama3', 'Llama-3', 'llama3', 'llama-3']):
                formatted_prompt = get_llama_without_answer(test_prompt)
                print(f"Using Llama format")
            elif 'Qwen' in hparams.model_name or 'qwen' in hparams.model_name:
                formatted_prompt = get_qwen_without_answer(test_prompt)
                print(f"Using Qwen format")
            else:
                formatted_prompt = test_prompt
                print(f"Using raw format")
        
            question = tokenizer([formatted_prompt], return_tensors='pt', padding=True)
        
            # Model-specific stop tokens
            generation_kwargs = {
                'input_ids': question['input_ids'].to('cuda'),
                'attention_mask': question['attention_mask'].to('cuda'),
                'do_sample': True,
                'temperature': 0.001,
                'max_new_tokens': 1024,
                'pad_token_id': tokenizer.pad_token_id,
            }
        
            # Qwen stop tokens
            if 'Qwen' in hparams.model_name or 'qwen' in hparams.model_name:
                # Get stop token IDs
                stop_token_ids = []
                for stop_str in ['<|im_end|>', '<|endoftext|>']:
                    stop_id = tokenizer.encode(stop_str, add_special_tokens=False)
                    if stop_id:
                        stop_token_ids.append(stop_id[0] if isinstance(stop_id, list) else stop_id)
                if stop_token_ids:
                    generation_kwargs['eos_token_id'] = stop_token_ids
                else:
                    generation_kwargs['eos_token_id'] = tokenizer.eos_token_id
            elif 'Llama' in hparams.model_name or 'llama' in hparams.model_name:
                # Llama stop tokens
                stop_token_ids = []
                for stop_str in ['<|eot_id|>', '<|end_of_text|>']:
                    stop_id = tokenizer.encode(stop_str, add_special_tokens=False)
                    if stop_id:
                        stop_token_ids.append(stop_id[0] if isinstance(stop_id, list) else stop_id)
                if stop_token_ids:
                    generation_kwargs['eos_token_id'] = stop_token_ids
                else:
                    generation_kwargs['eos_token_id'] = tokenizer.eos_token_id
            else:
                generation_kwargs['eos_token_id'] = tokenizer.eos_token_id
        
            with torch.no_grad():
                generated_ids = model.generate(**generation_kwargs)
        
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(question['input_ids'], generated_ids)
            ]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            data['original_prediction'] = output[0]
            gen_time = time() - gen_start
            print(f"[OK] Generation complete: {gen_time:.2f}s")
        
            print(f"\nComplete LLM Output:")
            print(f"{'-'*60}")
            print(output[0])
            print(f"{'-'*60}")
            
            # Step 3: Test generated code
            test_start = time()
            
            # Extract code from LLM output
            if dataset_type == "rustevo":
                generated_code = extract_rust_code(data.get('original_prediction', ''))
            else:  # pyevo → extract python code
                generated_code = extract_python_code(data.get('original_prediction', ''))
            print(f"\n📦 Extracted Code:")
            print(f"{'-'*60}")
            print(generated_code)
            print(f"{'-'*60}\n")
        
            if not generated_code or len(generated_code.strip()) < 10:
                data['test_status'] = 'EXTRACTION_FAILED'
                data['test_error'] = 'Failed to extract valid code'
                data['extracted_code'] = generated_code
                test_results['extraction_error'] += 1
                test_results['failed_samples'].append({
                    'id': sample_id, 'status': 'EXTRACTION_FAILED', 'error': 'Code extraction failed'
                })
                print(f"[FAIL] Failed: Code extraction failed")

                _tp = data.get('test_program', '')
                if dataset_type == 'rustevo':
                    _n = len(re.findall(r'#\[test\]', _tp)) or 1
                elif dataset_type == 'pyevo':
                    _n = len(re.findall(r'def\s+test_', _tp)) or 1
                else:
                    _n = 0
                test_results['test_cases_total'] += _n

                if alg_name == "GRACE":
                    restore_grace_model(model, weights_copy)
                elif alg_name == "AGRACE":
                    restore_agrace_model(model, weights_copy)
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            if not k.startswith('_'):
                                nethook.get_parameter(model, k)[...] = v.to("cuda")
                edited_data.append(data)
                continue
            
            if dataset_type == "rustevo":
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
                                'id': sample_id, 'status': 'SIGNATURE_ERROR', 'error': 'Incorrect function signature'
                            })
                            print(f"[FAIL] Failed: Incorrect function signature")

                            _n = len(re.findall(r'#\[test\]', data.get('test_program', ''))) or 1
                            test_results['test_cases_total'] += _n

                            if alg_name == "GRACE":
                                restore_grace_model(model, weights_copy)
                            elif alg_name == "AGRACE":
                                restore_agrace_model(model, weights_copy)
                            else:
                                with torch.no_grad():
                                    for k, v in weights_copy.items():
                                        nethook.get_parameter(model, k)[...] = v.to("cuda")
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
                                'id': sample_id, 'status': 'API_ERROR', 'error': error_msg
                            })
                            print(f"[FAIL] Failed: {error_msg}")

                            _n = len(re.findall(r'#\[test\]', data.get('test_program', ''))) or 1
                            test_results['test_cases_total'] += _n

                            if alg_name == "GRACE":
                                restore_grace_model(model, weights_copy)
                            elif alg_name == "AGRACE":
                                restore_agrace_model(model, weights_copy)
                            else:
                                with torch.no_grad():
                                    for k, v in weights_copy.items():
                                        nethook.get_parameter(model, k)[...] = v.to("cuda")
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
                
                    n_rust_cases = len(re.findall(r'#\[test\]', data.get('test_program', ''))) or 1
                    test_results['test_cases_total'] += n_rust_cases
                    rust_out = (result.get('stdout') or '') + (result.get('stderr') or '')
                    m_rust = re.search(r'test result:.*?(\d+) passed;\s*(\d+) failed', rust_out)
                    if m_rust:
                        _tc_p = int(m_rust.group(1))
                        test_results['test_cases_passed'] += min(_tc_p, n_rust_cases)
                    elif result['success']:
                        test_results['test_cases_passed'] += n_rust_cases

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
                        if result.get('error'):
                            print(f"   Error: {result['error'][:300]}")
                        if result.get('stderr'):
                            stderr_preview = result['stderr'][:500]
                            print(f"   Stderr:\n{stderr_preview}")

                    test_time = time() - test_start
                    print(f"[OK] Test complete: {test_time:.2f}s")
            
            elif dataset_type == "pyevo":
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
                        if not py_check_signature(generated_code, function_signature):
                            data['test_status'] = 'SIGNATURE_ERROR'
                            data['test_passed'] = False
                            data['test_error'] = f'Incorrect function signature: {function_signature}'
                            data['extracted_code'] = generated_code
                            test_results['signature_error'] += 1
                            test_results['failed_samples'].append({
                                'id': sample_id, 'status': 'SIGNATURE_ERROR',
                                'error': 'Incorrect function signature'
                            })
                            print(f"[FAIL] Failed: Incorrect function signature")

                            _n = len(re.findall(r'def\s+test_', data.get('test_program', ''))) or 1
                            test_results['test_cases_total'] += _n

                            if alg_name == "GRACE":
                                restore_grace_model(model, weights_copy)
                            elif alg_name == "AGRACE":
                                restore_agrace_model(model, weights_copy)
                            else:
                                with torch.no_grad():
                                    for k, v in weights_copy.items():
                                        nethook.get_parameter(model, k)[...] = v.to("cuda")
                            edited_data.append(data)
                            continue

                    api_name = data.get('name', '')
                    change_type = data.get('change_type', '')
                    api_module = data.get('module', '')
                    if api_name:
                        if not py_check_api_usage(generated_code, api_name, change_type, api_module):
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
                                'id': sample_id, 'status': 'API_ERROR', 'error': error_msg
                            })
                            print(f"[FAIL] Failed: {error_msg}")

                            _n = len(re.findall(r'def\s+test_', data.get('test_program', ''))) or 1
                            test_results['test_cases_total'] += _n

                            if alg_name == "GRACE":
                                restore_grace_model(model, weights_copy)
                            elif alg_name == "AGRACE":
                                restore_agrace_model(model, weights_copy)
                            else:
                                with torch.no_grad():
                                    for k, v in weights_copy.items():
                                        nethook.get_parameter(model, k)[...] = v.to("cuda")
                            edited_data.append(data)
                            continue

                    to_version = data.get('to_version', '')
                    python_bin = DEFAULT_PYTHON_BIN
                    try:
                        with ThreadPoolExecutor(max_workers=1) as _exec:
                            _fut = _exec.submit(get_sandbox_python, api_module, to_version)
                            try:
                                python_bin = _fut.result(timeout=30) or DEFAULT_PYTHON_BIN
                            except (FuturesTimeout, Exception):
                                _fut.cancel()
                                python_bin = DEFAULT_PYTHON_BIN
                    except Exception:
                        python_bin = DEFAULT_PYTHON_BIN

                    result = run_python_test(
                        generated_code,
                        test_program,
                        timeout=PY_DEFAULT_TIMEOUT,
                        python_bin=python_bin,
                    )

                    data['test_status'] = 'passed' if result['success'] else result.get('error', 'failed')
                    data['test_passed'] = result['success']
                    data['extracted_code'] = generated_code
                    if result.get('error'):
                        data['test_error'] = result['error']
                    if result.get('output'):
                        data['test_output'] = result['output']

                    py_test_prog = data.get('test_program', '')
                    n_py_cases = len(re.findall(r'def\s+test_', py_test_prog)) or 1
                    test_results['test_cases_total'] += n_py_cases

                    py_out = result.get('output', '') or ''
                    m_py = re.search(r'(\d+) passed(?:,\s*(\d+) failed)?', py_out)

                    if result['success']:
                        test_results['passed'] += 1
                        if m_py:
                            _tc_p = int(m_py.group(1))
                            test_results['test_cases_passed'] += min(_tc_p, n_py_cases)
                        else:
                            test_results['test_cases_passed'] += n_py_cases
                        print(f"[PASS] Test passed")
                    else:
                        if m_py:
                            _tc_p = int(m_py.group(1))
                            test_results['test_cases_passed'] += min(_tc_p, n_py_cases)
                        error_type = result.get('error', 'other')
                        if error_type == 'timeout':
                            test_results['timeout'] += 1
                        elif error_type == 'test_failed':
                            test_results['test_failed'] += 1
                        else:
                            test_results['other_error'] += 1
                        test_results['failed_samples'].append({
                            'id': sample_id, 'status': data['test_status'],
                            'error': result.get('error', '')
                        })
                        print(f"[FAIL] Test failed: {result.get('error', 'unknown')}")
                        if result.get('output'):
                            print(f"   Output: {result['output'][:300]}")

                    test_time = time() - test_start
                    print(f"[OK] Test complete: {test_time:.2f}s")


        
            if alg_name == "GRACE":
                restore_grace_model(model, weights_copy)
            elif alg_name == "AGRACE":
                restore_agrace_model(model, weights_copy)
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        if not k.startswith('_'):
                            nethook.get_parameter(model, k)[...] = v.to("cuda")
            
            edited_data.append(data)
        
        finally:
            torch.cuda.empty_cache()
            
            if (sample_idx + 1) % 10 == 0 or sample_idx == len(ds) - 1:
                tested = sample_idx + 1 - test_results['no_test']
                failed = tested - test_results['passed']

                snapshot = {
                    'sample_count': sample_idx + 1,
                    'tested': tested,
                    'passed': test_results['passed'],
                    'failed': failed,
                }
                


            print(f"\n  k={k}:")
            print(f"    ★ UPass@{k}:    {upass_k:6.2f}%")
            print(f"    pass@{k}(new):  {pass_k_new:6.2f}%")

        print(f"\n{'─'*50}")
        print(f"  Raw counts:")
        print(f"    Exclusive (UPass): {c_new_excl}/{tested_count} ({c_new_excl/tested_count*100:.2f}%)")
        print(f"    Inclusive:         {test_results['inclusive_pass']}/{tested_count} ({test_results['inclusive_pass']/tested_count*100:.2f}%)")
        print(f"{'─'*50}")

        print(f"\nNote: SPass@k (Specificity) HumanEval evaluate")


    else:
        dataset_label = "PyEvo" if dataset_type == "pyevo" else "RustEvo"
        if tested_count > 0:
            c_passed = test_results['passed']
            api_error_count = test_results['api_error'] + test_results['signature_error']
            aua_correct = tested_count - api_error_count
            aua_pct = aua_correct / tested_count * 100
            tc_passed = test_results['test_cases_passed']
            tc_total = test_results['test_cases_total']
            coverage_pct = tc_passed / tc_total * 100 if tc_total > 0 else 0.0

            print(f"\n{'─'*50}")
            print(f"[Stats] {dataset_label} Core Metrics")
            print(f"{'─'*50}")

            for k in [1, 2, 5]:
                if k > tested_count:
                    continue
                pass_k = pass_at_k(tested_count, c_passed, k) * 100
                print(f"\n  Pass@{k}:    {pass_k:6.2f}%")

            print(f"\n  AUA:        {aua_pct:6.2f}%  ({aua_correct}/{tested_count})")
            print(f"  Coverage:   {coverage_pct:6.2f}%  ({tc_passed}/{tc_total} test cases)")
            print(f"\n[PASS] passed: {c_passed}/{tested_count} ({c_passed/tested_count*100:.2f}%)")
        else:
            print(f"\n[PASS] Passed: 0")
            print(f"\n[FAIL] Failed: 0")

    if failed_count > 0:
        print(f"\n[FAIL] Error breakdown:")
        if test_results['compilation_failed'] > 0:
            comp_rate = test_results['compilation_failed']/tested_count*100 if tested_count > 0 else 0
            print(f" - error: {test_results['compilation_failed']} ({comp_rate:.1f}%)")
        if test_results['test_failed'] > 0:
            test_rate = test_results['test_failed']/tested_count*100 if tested_count > 0 else 0
            print(f" - testfailed: {test_results['test_failed']} ({test_rate:.1f}%)")
        if test_results['timeout'] > 0:
            timeout_rate = test_results['timeout']/tested_count*100 if tested_count > 0 else 0
            print(f" - Test timeout: {test_results['timeout']} ({timeout_rate:.1f}%)")
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
    
    stats_summary = {
        'total': test_results['total'],
        'tested': tested_count,
        'passed': test_results['passed'],
        'progress': progress_snapshots,
        'failed': failed_count,
    }
    

    print(f"\nResults saved to: {output_file}")

    summary = {
        'algorithm': alg_name,
        'model': hparams.model_name,
        'dataset': ds_name,
        'use_aeg': alg_name in aeg_methods,
        'has_graphs': prebuilt_graphs is not None,
        'tested_count': tested_count,
        **test_results
    }


    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AEG on RustEvo")
    
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit", "AlphaEdit_ARE", "AlphaEdit_ARE_AEG", "UnKE_ARE", "UnKE_ARE_AEG", "MEMIT_ARE", "MEMIT_ARE_AEG", "ROME", "GRACE", "AGRACE", "STAR"],
        default="AlphaEdit_ARE_AEG",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Qwen2.5-7B-Instruct.json",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        default="rustevo",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        default="./data/rustevo_graphs",
        help="Directory containing prebuilt graphs",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
    )
    
    args = parser.parse_args()
    
    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.graph_dir,
        args.num_edits,
    )
