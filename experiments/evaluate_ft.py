"""
FT-L / LoRA / AdaLoRA evaluation for RustEvo / PyEvo (edit-like protocol).

Protocol per sample:
1) Restore base trainable weights (or reset adapters for LoRA/AdaLoRA)
2) One-sample update (batch=1) using rephrased_query -> code
3) Generate on original query
4) Run dataset-specific tests (same style as evaluate_gnn.py)

Metrics: Pass@1/2/5, AUA, Coverage

This matches model-edit's "one edit -> one output" comparison style.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, AdaLoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft library not available. Install with: pip install peft")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Pass@k: 1 - C(n-c, k) / C(n, k)"""
    if n == 0:
        return 0.0
    if c > n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsets import RustEvoDataset, PyEvoDataset
from util.rust_cargo_test import (
    DEFAULT_TEST_TIMEOUT,
    check_api_usage,
    check_function_signature,
    prepare_rust_test_inputs,
    run_rust_test_auto,
)
from util.python_pytest_test import (
    run_python_test,
    check_function_signature as py_check_signature,
    check_api_usage as py_check_api_usage,
    get_sandbox_python,
    DEFAULT_PYTHON_BIN,
    DEFAULT_TEST_TIMEOUT as PY_DEFAULT_TIMEOUT,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_pad_token(model, tok):
    if tok.pad_token is None:
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tok))


def parse_layer_idx(name: str):
    for key in [".layers.", ".h."]:
        if key in name:
            rest = name.split(key, 1)[1]
            idx = rest.split(".", 1)[0]
            if idx.isdigit():
                return int(idx)
    return None


def parse_layer_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return sorted(set(out))


def setup_ftl_trainable(
    model,
    unfreeze_last_n_layers: int,
    train_lm_head: bool,
    explicit_layers: List[int],
):
    for p in model.parameters():
        p.requires_grad = False

    max_idx = -1
    for n, _ in model.named_parameters():
        idx = parse_layer_idx(n)
        if idx is not None:
            max_idx = max(max_idx, idx)
    if max_idx < 0:
        raise RuntimeError("Cannot infer transformer layers for FT-L")

    explicit_set = set(explicit_layers or [])
    if explicit_set:
        selected_desc = f"explicit_layers={sorted(explicit_set)}"
    else:
        th = max_idx - unfreeze_last_n_layers + 1
        selected_desc = f"last_n_layers={unfreeze_last_n_layers} (idx>={th})"

    trainable_names: List[str] = []
    for n, p in model.named_parameters():
        idx = parse_layer_idx(n)
        if idx is not None and ((idx in explicit_set) if explicit_set else (idx >= th)):
            p.requires_grad = True
            trainable_names.append(n)
        elif train_lm_head and ("lm_head" in n or n.endswith("model.norm.weight")):
            p.requires_grad = True
            trainable_names.append(n)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_names, trainable, total, selected_desc


def setup_lora(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
):
    """Setup LoRA adapters for the model"""
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft library is required for LoRA. Install with: pip install peft")
    
    # Default target modules for common architectures
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return model, trainable, total, f"r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}"


def setup_adalora(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    adalora_init_r: int = 12,
    target_modules: List[str] = None,
    total_step: int = 1,
):
    """Setup AdaLoRA adapters for the model"""
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft library is required for AdaLoRA. Install with: pip install peft")
    
    # Default target modules for common architectures
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = AdaLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_r=lora_r,
        init_r=adalora_init_r,
        tinit=0,
        tfinal=0,
        deltaT=1,
        total_step=total_step,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return model, trainable, total, f"r={lora_r}, alpha={lora_alpha}, init_r={adalora_init_r}"


def snapshot_trainable(model, trainable_names: List[str]) -> Dict[str, torch.Tensor]:
    out = {}
    named = dict(model.named_parameters())
    for n in trainable_names:
        out[n] = named[n].detach().cpu().clone()
    return out


def restore_trainable(model, trainable_names: List[str], snapshot: Dict[str, torch.Tensor], device: str):
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n in trainable_names:
            named[n].copy_(snapshot[n].to(named[n].device))


def reset_lora_adapters(model):
    """Reset LoRA/AdaLoRA adapter weights to initial state"""
    if isinstance(model, PeftModel):
        # Reset adapter parameters
        for name, param in model.named_parameters():
            if "lora_" in name or "ranknum" in name:
                param.data.zero_()
                if "lora_A" in name:
                    # Re-initialize lora_A with kaiming uniform
                    torch.nn.init.kaiming_uniform_(param.data, a=5**0.5)
                elif "lora_B" in name:
                    # lora_B stays as zeros
                    pass


def get_input_device(model) -> torch.device:
    # Works for both single-device and device_map sharded models
    return next(model.parameters()).device


def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""


def format_train_pair(sample: dict, model_name: str, ds_name: str = "rustevo") -> Tuple[str, str]:
    """return (prompt_text, answer_text) FT-L training"""
    if ds_name == "pyevo":
        prompt = build_pyevo_prompt(sample, use_rephrased=True)
    else:
        prompt = build_rust_prompt(sample, use_rephrased=True)
    code = sample.get("code", "")
    if any(name in model_name for name in ["Llama3", "Llama-3", "llama3", "llama-3"]):
        return get_llama_without_answer(prompt), code + "<|eot_id|>"
    if "Qwen" in model_name or "qwen" in model_name:
        return get_qwen_without_answer(prompt), code + "<|im_end|>"
    return prompt + "\n", code


def build_rust_prompt(sample: dict, use_rephrased: bool = False) -> str:
    """Build code generation prompt for RustEvo."""
    query = sample.get("rephrased_query") if use_rephrased else sample.get("query", "")
    function_signature = sample.get("function_signature", "")
    name = sample.get("name", "")
    module = sample.get("module", "")
    is_crate = not (
        module.startswith("std::") or module.startswith("core::") or module.startswith("alloc::")
    )

    if not is_crate:
        return f"""You are an expert Rust programmer. Implement the following Rust function:

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

Respond with ONLY the Rust function implementation."""

    crate_name = module.split("::")[0] if "::" in module else module
    return f"""You are an expert Rust programmer. Implement the following Rust function:

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

Respond with ONLY the Rust function implementation."""


def build_pyevo_prompt(sample: dict, use_rephrased: bool = False) -> str:
    """Build code generation prompt for PyEvo."""
    query = sample.get("rephrased_query") if use_rephrased else sample.get("query", "")
    name               = sample.get("name", "")
    module             = sample.get("module", "")
    function_signature = sample.get("function_signature", "").strip()

    return f"""You are an expert Python programmer. Implement the following Python function.

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

Respond with ONLY the Python function implementation."""


def extract_rust_code(text: str) -> str:
    matches = re.findall(r"```(?:rust)?\s*(.*?)```", text, re.DOTALL)
    if matches:
        valid = [m.strip() for m in matches if m.strip()]
        if valid:
            return max(valid, key=len)
    return text.strip("` \n\t")


def extract_python_code(text: str) -> str:
    """ LLM responseextract Python code(filtertestfunction)"""
    code = None
    if "```python" in text:
        start = text.find("```python") + len("```python\n")
        end = text.find("```", start)
        if end != -1:
            code = text[start:end].strip()
    if code is None and "[PYTHON]" in text and "[/PYTHON]" in text:
        start = text.find("[PYTHON]") + len("[PYTHON]")
        end = text.find("[/PYTHON]")
        if end > start:
            code = text[start:end].strip()
    if code is None and "def " in text:
        code = text
    if code is None:
        return text.strip()

    lines = code.split("\n")
    result_lines = []
    in_test_func = False
    main_func_indent = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if result_lines and not in_test_func:
                result_lines.append(line)
            continue
        if stripped.startswith("def test_"):
            in_test_func = True
            continue
        if in_test_func:
            if line.startswith((" ", "\t")):
                continue
            else:
                in_test_func = False
        if stripped.startswith("test_") and "(" in stripped:
            continue
        if stripped.startswith("#") and "test" in stripped.lower():
            continue
        if stripped.startswith(("import ", "from ")):
            result_lines.append(line)
            continue
        if stripped.startswith("def ") and not stripped.startswith("def test_"):
            main_func_indent = len(line) - len(line.lstrip())
            result_lines.append(line)
            continue
        if main_func_indent is not None:
            current_indent = len(line) - len(line.lstrip()) if stripped else 0
            if current_indent > main_func_indent or not stripped:
                result_lines.append(line)
            elif stripped.startswith("def ") and not stripped.startswith("def test_"):
                main_func_indent = len(line) - len(line.lstrip())
                result_lines.append(line)

    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    return "\n".join(result_lines) if result_lines else text.strip()


def get_stop_tokens(tok, model_name: str):
    if "Qwen" in model_name or "qwen" in model_name:
        cands = ["<|im_end|>", "<|endoftext|>"]
    elif "Llama" in model_name or "llama" in model_name:
        cands = ["<|eot_id|>", "<|end_of_text|>"]
    else:
        return tok.eos_token_id
    out = []
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            out.append(ids[0] if isinstance(ids, list) else ids)
    return out or tok.eos_token_id


def one_sample_update(
    model,
    tok,
    sample: dict,
    model_name: str,
    device: str,
    lr: float,
    n_steps: int,
    max_length: int,
    ds_name: str = "rustevo",
    method: str = "FT-L",
):
    """Single sample update for FT-L / LoRA / AdaLoRA"""
    prompt_text, answer_text = format_train_pair(sample, model_name, ds_name)
    full_text = prompt_text + answer_text
    enc = tok(
        [full_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    in_dev = get_input_device(model)
    input_ids = enc["input_ids"].to(in_dev)
    attention_mask = enc["attention_mask"].to(in_dev)
    labels = input_ids.clone()
    # Optimize on full prompt+answer sequence.
    labels[attention_mask == 0] = -100

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(max(1, n_steps)):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def _generate_code(model, tok, model_name: str, prompt: str,
                   do_sample: bool, temperature: float) -> str:
    """callmodelgeneratecode, returnoriginal LLM Output"""
    if any(name in model_name for name in ["Llama3", "Llama-3", "llama3", "llama-3"]):
        formatted = get_llama_without_answer(prompt)
    elif "Qwen" in model_name or "qwen" in model_name:
        formatted = get_qwen_without_answer(prompt)
    else:
        formatted = prompt

    q = tok([formatted], return_tensors="pt", padding=True, add_special_tokens=False)
    in_dev = get_input_device(model)
    kwargs = {
        "input_ids": q["input_ids"].to(in_dev),
        "attention_mask": q["attention_mask"].to(in_dev),
        "do_sample": do_sample,
        "max_new_tokens": 512,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": get_stop_tokens(tok, model_name),
    }
    if do_sample:
        kwargs["temperature"] = temperature
    model.eval()
    with torch.no_grad():
        out_ids = model.generate(**kwargs)
    out_ids = [o[len(i):] for i, o in zip(q["input_ids"], out_ids)]
    return tok.batch_decode(out_ids, skip_special_tokens=True)[0]


def _parse_rust_test_output(stdout: str) -> Tuple[int, int]:
    """Parse Rust testOutput, return (passed, total)"""
    if not stdout:
        return 0, 0
    m = re.search(r"test result:.*?(\d+) passed; (\d+) failed", stdout)
    if m:
        p, f = int(m.group(1)), int(m.group(2))
        return p, p + f
    return 0, 0


def _parse_pytest_output(output: str) -> Tuple[int, int]:
    """Parse pytest Output, return (passed, total)"""
    if not output:
        return 0, 0
    mp = re.search(r"(\d+) passed", output)
    mf = re.search(r"(\d+) failed", output)
    me = re.search(r"(\d+) error", output)
    p = int(mp.group(1)) if mp else 0
    f = (int(mf.group(1)) if mf else 0) + (int(me.group(1)) if me else 0)
    return p, p + f


def _count_rust_tests(test_program: str) -> int:
    return len(re.findall(r"#\[test\]", test_program or ""))


def _count_py_tests(test_program: str) -> int:
    return len(re.findall(r"def test_", test_program or ""))


def evaluate_one_sample_rust(
    model,
    tok,
    model_name: str,
    data: dict,
    do_sample: bool = True,
    temperature: float = 0.001,
) -> dict:
    """RustEvo sampleevaluate, returnwith test_cases_passed/total results"""
    pred = _generate_code(model, tok, model_name,
                          build_rust_prompt(data, use_rephrased=False),
                          do_sample, temperature)
    result = {
        "id": data.get("id"),
        "original_prediction": pred,
        "test_status": None,
        "test_passed": False,
        "test_cases_passed": 0,
        "test_cases_total": 0,
    }
    n_source = _count_rust_tests(data.get("test_program", ""))

    code = extract_rust_code(pred)
    result["extracted_code"] = code
    if not code or len(code.strip()) < 10:
        result["test_status"] = "EXTRACTION_FAILED"
        result["test_cases_total"] = n_source
        return result

    test_program = data.get("test_program", "")
    if not test_program or test_program == "INCORRECT CODE":
        result["test_status"] = "NO_TEST"
        return result

    sig = data.get("function_signature", "")
    if sig and not check_function_signature(code, sig):
        result["test_status"] = "SIGNATURE_ERROR"
        result["test_cases_total"] = n_source
        return result

    api_name    = data.get("name", "")
    change_type = data.get("change_type", "")
    api_module  = data.get("module", "")
    replacement_api = data.get("replacement_api", "")
    if api_name and not check_api_usage(code, api_name, change_type, api_module,
                                        test_program, replacement_api):
        result["test_status"] = "API_ERROR"
        result["test_cases_total"] = n_source
        return result

    to_version = data.get("to_version", "1.84.0")
    clean_code, clean_test, rust_version, crate_version, _, _ = prepare_rust_test_inputs(
        code, test_program, api_module, to_version, dedup_imports=True,
    )
    tr = run_rust_test_auto(
        clean_code, clean_test, api_module, rust_version, crate_version,
        timeout=DEFAULT_TEST_TIMEOUT,
    )
    result["test_status"]  = tr.get("status")
    result["test_passed"]  = bool(tr.get("success", False))
    result["test_stdout"]  = tr.get("stdout", "")
    result["test_stderr"]  = tr.get("stderr", "")
    if tr.get("error"):
        result["test_error"] = tr["error"]

    tc_p, _ = _parse_rust_test_output(tr.get("stdout", ""))
    result["test_cases_passed"] = min(tc_p, n_source)
    result["test_cases_total"]  = n_source
    return result


def evaluate_one_sample_pyevo(
    model,
    tok,
    model_name: str,
    data: dict,
    do_sample: bool = True,
    temperature: float = 0.001,
    test_python: str = DEFAULT_PYTHON_BIN,
    use_sandbox: bool = False,
    sandbox_cache: str = "data/sandbox_cache",
) -> dict:
    """PyEvo sampleevaluate, returnwith test_cases_passed/total results"""
    pred = _generate_code(model, tok, model_name,
                          build_pyevo_prompt(data, use_rephrased=False),
                          do_sample, temperature)
    result = {
        "id": data.get("id"),
        "original_prediction": pred,
        "test_status": None,
        "test_passed": False,
        "test_cases_passed": 0,
        "test_cases_total": 0,
    }
    n_source = _count_py_tests(data.get("test_program", ""))

    code = extract_python_code(pred)
    result["extracted_code"] = code
    if not code or len(code.strip()) < 10:
        result["test_status"] = "EXTRACTION_FAILED"
        result["test_cases_total"] = n_source
        return result

    test_program = data.get("test_program", "")
    if not test_program:
        result["test_status"] = "NO_TEST"
        return result

    function_signature = data.get("function_signature", "")
    if function_signature and not py_check_signature(code, function_signature):
        result["test_status"] = "SIGNATURE_ERROR"
        result["test_cases_total"] = n_source
        return result

    api_name    = data.get("name", "")
    change_type = data.get("change_type", "")
    api_module  = data.get("module", "")
    if api_name and not py_check_api_usage(code, api_name, change_type, api_module):
        result["test_status"] = "API_ERROR"
        result["test_cases_total"] = n_source
        return result

    actual_python = test_python
    if use_sandbox and api_module and data.get("to_version"):
        sb_py = get_sandbox_python(
            api_module.split(".")[0], data["to_version"], sandbox_cache
        )
        if sb_py:
            actual_python = sb_py

    clean_code = code
    clean_test = test_program
    tr = run_python_test(clean_code, clean_test, timeout=PY_DEFAULT_TIMEOUT,
                         python_bin=actual_python)
    result["test_status"] = "PASSED" if tr["success"] else "FAILED"
    result["test_passed"] = tr["success"]
    result["test_output"] = tr.get("output", "")
    if tr.get("error"):
        result["test_error"] = tr["error"]

    tc_p, _ = _parse_pytest_output(tr.get("output", ""))
    result["test_cases_passed"] = min(tc_p, n_source)
    result["test_cases_total"]  = n_source
    return result


def main():
    parser = argparse.ArgumentParser("FT-L / LoRA / AdaLoRA evaluate on RustEvo / PyEvo")
    # Method selection
    parser.add_argument("--method", type=str, default="FT-L",
                        choices=["FT-L", "LoRA", "AdaLoRA"],
                        help="Fine-tuning method: FT-L, LoRA, or AdaLoRA")
    parser.add_argument("--ds_name", type=str, default="rustevo",
                        choices=["rustevo", "pyevo"], help="dataname")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_size_limit", type=int, default=-1)
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device_map", type=str, default="single", choices=["single", "auto"])
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    
    # FT-L specific parameters
    parser.add_argument("--ftl_unfreeze_last_n_layers", type=int, default=4)
    parser.add_argument("--ftl_unfreeze_layers", type=str, default="")
    parser.add_argument("--ftl_train_lm_head", action="store_true")
    
    # LoRA/AdaLoRA specific parameters
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--adalora_init_r", type=int, default=12,
                        help="AdaLoRA initial rank")
    parser.add_argument("--lora_target_modules", type=str, default="",
                        help="Comma-separated list of target modules for LoRA")
    
    # Training parameters
    parser.add_argument("--edit_lr", type=float, default=5e-6)
    parser.add_argument("--edit_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_prefix", type=str, default="FTL")
    parser.add_argument("--eval_do_sample", action="store_true")
    parser.add_argument("--eval_temperature", type=float, default=0.001)
    parser.add_argument("--sandbox", action="store_true",
                        help="[PyEvo] versionsandboxtest")
    parser.add_argument("--sandbox_cache",
                        default="data/sandbox_cache",
                        help="[PyEvo] sandboxdirectory")
    parser.add_argument("--test_python", default=DEFAULT_PYTHON_BIN,
                        help="[PyEvo] pytest Use Python ")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    load_kwargs = dict(trust_remote_code=True, torch_dtype=dtype_map[args.torch_dtype])
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    ensure_pad_token(model, tok)
    if args.device_map == "single":
        model.to(device)

    DATA_DIR = "data"
    ds_size = None if args.dataset_size_limit is None or args.dataset_size_limit < 0 else args.dataset_size_limit
    if args.ds_name == "pyevo":
        ds = PyEvoDataset(DATA_DIR, model_name=model.config._name_or_path, size=ds_size)
    else:
        ds = RustEvoDataset(DATA_DIR, model_name=model.config._name_or_path, size=ds_size)

    # Setup method-specific training configuration
    trainable_names = None
    base_snapshot = None
    explicit_layers = None
    
    if args.method == "FT-L":
        explicit_layers = parse_layer_list(args.ftl_unfreeze_layers)
        trainable_names, trn, ttl, layer_select_desc = setup_ftl_trainable(
            model,
            unfreeze_last_n_layers=args.ftl_unfreeze_last_n_layers,
            train_lm_head=args.ftl_train_lm_head,
            explicit_layers=explicit_layers,
        )
        print(f"FT-L trainable params: {trn}/{ttl} ({100.0*trn/ttl:.2f}%)")
        print(f"FT-L layer selection: {layer_select_desc}")
        base_snapshot = snapshot_trainable(model, trainable_names)
    
    elif args.method == "LoRA":
        target_modules = None
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        model, trn, ttl, lora_desc = setup_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        print(f"LoRA trainable params: {trn}/{ttl} ({100.0*trn/ttl:.2f}%)")
        print(f"LoRA config: {lora_desc}")
    
    elif args.method == "AdaLoRA":
        target_modules = None
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        model, trn, ttl, adalora_desc = setup_adalora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            adalora_init_r=args.adalora_init_r,
            target_modules=target_modules,
            total_step=max(1, args.edit_steps),
        )
        print(f"AdaLoRA trainable params: {trn}/{ttl} ({100.0*trn/ttl:.2f}%)")
        print(f"AdaLoRA config: {adalora_desc}")

    stats = {
        "total": len(ds),
        "passed": 0,
        "no_test": 0,
        "compilation_failed": 0,
        "test_failed": 0,
        "timeout": 0,
        "signature_error": 0,
        "api_error": 0,
        "extraction_error": 0,
        "other_error": 0,
        "test_cases_passed": 0,
        "test_cases_total": 0,
    }
    raw_status_counter = Counter()
    results = []
    start = time()

    ds_label = "PyEvo" if args.ds_name == "pyevo" else "RustEvo"
    for i in tqdm(range(len(ds)), desc=f"{args.method} eval [{ds_label}]"):
        sample = ds[i]

        # Reset model state for each sample based on method
        if args.method == "FT-L":
            restore_trainable(model, trainable_names, base_snapshot, device)
        else:  # LoRA or AdaLoRA
            reset_lora_adapters(model)

        one_sample_update(
            model, tok, sample, model.config._name_or_path, device,
            lr=args.edit_lr, n_steps=args.edit_steps,
            max_length=args.max_length, ds_name=args.ds_name,
            method=args.method,
        )

        if args.ds_name == "pyevo":
            r = evaluate_one_sample_pyevo(
                model, tok, model.config._name_or_path, sample,
                do_sample=args.eval_do_sample, temperature=args.eval_temperature,
                test_python=args.test_python,
                use_sandbox=args.sandbox, sandbox_cache=args.sandbox_cache,
            )
        else:
            r = evaluate_one_sample_rust(
                model, tok, model.config._name_or_path, sample,
                do_sample=args.eval_do_sample, temperature=args.eval_temperature,
            )

        status = r.get("test_status")
        raw_status_counter[str(status) if status is not None else "<NONE>"] += 1

        if r.get("test_passed"):
            stats["passed"] += 1
        else:
            if status == "NO_TEST":
                stats["no_test"] += 1
            elif status == "SIGNATURE_ERROR":
                stats["signature_error"] += 1
            elif status == "API_ERROR":
                stats["api_error"] += 1
            elif status == "EXTRACTION_FAILED":
                stats["extraction_error"] += 1
            elif status and "COMPILE" in status.upper():
                stats["compilation_failed"] += 1
            elif status and "TIMEOUT" in status.upper():
                stats["timeout"] += 1
            elif status and ("TEST" in status.upper() or status == "FAILED"):
                stats["test_failed"] += 1
            else:
                stats["other_error"] += 1

        if status != "NO_TEST":
            stats["test_cases_passed"] += r.get("test_cases_passed", 0)
            stats["test_cases_total"]  += r.get("test_cases_total", 0)

        merged = dict(sample)
        merged.update(r)
        results.append(merged)

        if (i + 1) % 10 == 0 or i == len(ds) - 1:
            processed = i + 1
            tested = max(0, processed - stats["no_test"])
            passed = stats["passed"]
            pass_rate = (passed / tested * 100.0) if tested > 0 else 0.0
            tc_p = stats["test_cases_passed"]
            tc_t = stats["test_cases_total"]
            cov = (tc_p / tc_t * 100.0) if tc_t > 0 else 0.0
            api_usage_err = stats["signature_error"] + stats["api_error"]
            aua = ((tested - api_usage_err) / tested * 100.0) if tested > 0 else 0.0
            print(
                f"[Progress] {processed}/{len(ds)} | "
                f"tested={tested} passed={passed} "
                f"Pass@1={pass_rate:.2f}% AUA={aua:.2f}% Coverage={cov:.2f}%"
            )

    elapsed = time() - start
    tested = max(0, stats["total"] - stats["no_test"])
    c_passed = stats["passed"]
    api_usage_err = stats["signature_error"] + stats["api_error"]
    tc_p = stats["test_cases_passed"]
    tc_t = stats["test_cases_total"]

    aua_pct  = (tested - api_usage_err) / tested * 100.0 if tested > 0 else 0.0
    cov_pct  = tc_p / tc_t * 100.0 if tc_t > 0 else 0.0

    pass_at_k_metrics: Dict[str, float] = {}
    for k in [1, 2, 5]:
        if tested >= k:
            pass_at_k_metrics[f"pass@{k}"] = pass_at_k(tested, c_passed, k) * 100.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.model_name).name
    result_file  = out_dir / f"{args.output_prefix}_{model_tag}_{args.ds_name}_result.json"
    summary_file = out_dir / f"{args.output_prefix}_{model_tag}_{args.ds_name}_summary.json"

    summary = {
        "algorithm": args.method,
        "dataset": args.ds_name,
        "model": model.config._name_or_path,
        "edit_protocol": "per-sample one-edit-one-output",
        "edit_steps": args.edit_steps,
        "edit_lr": args.edit_lr,
        "total_samples": stats["total"],
        "tested_samples": tested,
        "metrics": {
            **pass_at_k_metrics,
            "AUA": round(aua_pct, 2),
            "Coverage": round(cov_pct, 2),
        },
        "tc_passed": tc_p,
        "tc_total": tc_t,
        "time_sec": round(elapsed, 1),
        "raw_status_counter": dict(raw_status_counter),
        **{k: v for k, v in stats.items()},
    }
    
    # Add method-specific parameters to summary
    if args.method == "FT-L":
        summary["ftl_unfreeze_last_n_layers"] = args.ftl_unfreeze_last_n_layers
        summary["ftl_unfreeze_layers"] = explicit_layers if args.ftl_unfreeze_layers else []
        summary["ftl_train_lm_head"] = args.ftl_train_lm_head
    elif args.method == "LoRA":
        summary["lora_r"] = args.lora_r
        summary["lora_alpha"] = args.lora_alpha
        summary["lora_dropout"] = args.lora_dropout
    elif args.method == "AdaLoRA":
        summary["lora_r"] = args.lora_r
        summary["lora_alpha"] = args.lora_alpha
        summary["adalora_init_r"] = args.adalora_init_r

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    failed = max(0, tested - c_passed)
    def pct(x: int) -> float:
        return (x / tested * 100.0) if tested > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"{ds_label} Test Results ({args.method})")
    print(f"{'='*55}")
    print(f"Total samples : {stats['total']}")
    print(f"Tested samples: {tested}")
    print(f"[PASS] Passed     : {c_passed}/{tested} ({pct(c_passed):.2f}%)")
    print(f"[FAIL] Failed     : {failed}/{tested} ({pct(failed):.2f}%)")
    print()

    if tested > 0:
        print(f"{'─'*55}")
        print(f"[Stats] Core Metrics")
        print(f"{'─'*55}")
        for k in [1, 2, 5]:
            if tested >= k:
                print(f"  Pass@{k}  : {pass_at_k(tested, c_passed, k)*100:6.2f}%")
        print(f"  AUA     : {aua_pct:6.2f}%  "
              f"(sig={stats['signature_error']}, api={stats['api_error']})")
        print(f"  Coverage: {cov_pct:6.2f}%  ({tc_p}/{tc_t})")
        print(f"{'─'*55}")

    print()
    print("[FAIL] Error breakdown:")
    if args.ds_name == "rustevo":
        print(f" error : {stats['compilation_failed']} ({pct(stats['compilation_failed']):.1f}%)")
    print(f" testfailed : {stats['test_failed']} ({pct(stats['test_failed']):.1f}%)")
    print(f" Function signature error : {stats['signature_error']} ({pct(stats['signature_error']):.1f}%)")
    print(f" APIUseerror : {stats['api_error']} ({pct(stats['api_error']):.1f}%)")
    print(f" Code extraction failed : {stats['extraction_error']} ({pct(stats['extraction_error']):.1f}%)")
    print(f" timeout : {stats['timeout']} ({pct(stats['timeout']):.1f}%)")
    print(f" Other error : {stats['other_error']} ({pct(stats['other_error']):.1f}%)")
    print()
    print(f"Model: {model_tag}")
    print(f"{'='*55}")
    print(f"Saved result : {result_file}")
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
