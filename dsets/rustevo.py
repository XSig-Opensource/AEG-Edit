import json
from pathlib import Path


def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""


class RustEvoDataset:
    """
    RustEvo Dataset loader for API evolution knowledge editing tasks.
    """

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        data_dir = Path(data_dir)

        if not data_dir.name == "RustEvo":
            data_dir = data_dir / "RustEvo"
        
        rustevo_file = data_dir / "RustEvo.json"

        if not rustevo_file.exists():
            raise FileNotFoundError(
                f"RustEvo data file not found: {rustevo_file}\n"
                f"Please ensure the file exists in {data_dir}\n"
                f"Hint: you can pass 'data' or 'data/RustEvo', the class handles both"
            )
        
        print(f"Loading RustEvo dataset: {rustevo_file}")
        with open(rustevo_file, 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        
        processed_data = []
        for sample in raw:
            processed = self._process_sample(sample, model_name)
            processed_data.append(processed)
        
        self._data = processed_data[:size] if size else processed_data
        print(f"Loaded {len(self._data)} samples")

    def _process_sample(self, sample: dict, model_name: str) -> dict:
        """Process a single sample and construct the editing format."""
        
        
        api_name = sample.get('name')
        api_module = sample.get('module')
        from_version = sample.get('from_version', 'unknown')
        to_version = sample.get('to_version', 'unknown')
        
        query = sample.get('rephrased_query') or sample.get('query', '')
        original_query = sample.get('query', '')
        
        signature = sample.get('signature', '')
        
        is_std_lib = api_module and any(api_module.startswith(prefix) for prefix in ['std::', 'core::', 'alloc::', 'std', 'core', 'alloc'])
        crate_name = api_module.split('::')[0] if api_module and '::' in api_module else api_module

        
        if api_name and api_module and query:
            api_info = f"{api_module}::{api_name}"
            func_sig = sample.get("function_signature", "").strip()
            api_signature = signature or f"{api_info}"
            
            subject_parts = [
                f"API Information: {api_info}",
                f"API Signature: {api_signature}",
                f"Task: {query}",
                "Function Signature:",
                f"```rust\n{func_sig}\n```",
            ]
            subject_raw = "\n".join(subject_parts)
            
            if any(name in model_name for name in ['Llama3', 'Llama-3', 'llama3', 'llama-3']):
                subject = get_llama_without_answer(subject_raw)
            elif 'Qwen' in model_name or 'qwen' in model_name:
                subject = get_qwen_without_answer(subject_raw)
            else:
                subject = subject_raw
            
            target = sample.get('code', '')
            if not target:
                target = sample.get('source_code', f"// Implementation for {api_name}")
        else:
            subject_raw = query if query else f"Implement using {api_module}::{api_name}"
            if any(name in model_name for name in ['Llama3', 'Llama-3', 'llama3', 'llama-3']):
                subject = get_llama_without_answer(subject_raw)
            elif 'Qwen' in model_name or 'qwen' in model_name:
                subject = get_qwen_without_answer(subject_raw)
            else:
                subject = subject_raw
            target = sample.get('code', sample.get('source_code', ''))
        
        question = subject
        
        if any(name in model_name for name in ['Llama3', 'Llama-3', 'llama3', 'llama-3']):
            answer = sample.get('code', '') + '<|eot_id|>'
        elif 'Qwen' in model_name or 'qwen' in model_name:
            answer = sample.get('code', '') + '<|im_end|>'
        else:
            answer = sample.get('code', '')
        
        sample_id = f"{api_name}_{from_version}_to_{to_version}" if api_name else "unknown"
        
        return {
            'id': sample_id,
            
            'subject': subject,
            'target': target,
            
            'question': question,
            'answer': answer,
            'para_question': question,
            'sub_question': [question],
            
            'test_program': sample.get('test_program', ''),
            'category': 'code',
            
            'module': api_module,
            'from_version': from_version,
            'to_version': to_version,
            
            'query': original_query,
            'rephrased_query': sample.get('rephrased_query', ''),
            'function_signature': sample.get('function_signature', ''),
            'name': api_name,
            
            'source_code': sample.get('source_code', ''),
            'old_source_code': sample.get('old_source_code', ''),
            'signature': sample.get('signature', ''),
            'documentation': sample.get('documentation', ''),
            'change_type': sample.get('change_type', 'signature'),
            'type': sample.get('type', 'fn'),
            'code': sample.get('code', ''),
        }

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
