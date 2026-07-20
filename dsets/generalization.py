"""
Generalization Dataset Loader — API test

samplewith API query-code-test :
  - edit_* : Model editing( API )
  - gen_* : test(Validate API)

return item with:
  - edit ( apply_algo)
  - gen (test prompt Buildtest)
  - API data
"""

import io
import json
import tokenize
from pathlib import Path


def strip_code_comments(code: str) -> str:
    """ Python code docstring"""
    if not code or not code.strip():
        return code
    try:
        import ast as _ast

        docstring_lines: set = set()
        try:
            tree = _ast.parse(code)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                     _ast.ClassDef, _ast.Module)):
                    if (node.body and
                            isinstance(node.body[0], _ast.Expr) and
                            isinstance(node.body[0].value, _ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        ds = node.body[0]
                        for ln in range(ds.lineno, ds.end_lineno + 1):
                            docstring_lines.add(ln)
        except SyntaxError:
            pass

        result_lines = []
        for lineno, line in enumerate(code.splitlines(), start=1):
            if lineno in docstring_lines:
                continue
            stripped = _strip_inline_comment(line)
            result_lines.append(stripped)

        cleaned: list = []
        prev_blank = False
        for line in result_lines:
            if line.strip() == '':
                if not prev_blank:
                    cleaned.append('')
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False

        return '\n'.join(cleaned).strip()
    except Exception:
        return code


def _strip_inline_comment(line: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
        for tok_type, _, tok_start, tok_end, _ in tokens:
            if tok_type == tokenize.COMMENT:
                return line[:tok_start[1]].rstrip()
        return line.rstrip()
    except tokenize.TokenError:
        return line.rstrip()


def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""


class GeneralizationDataset:
    """
    testdata — samplewith API task

    return item editing pipeline (subject/target/question/answer)
    test pipeline (gen_query/gen_function_signature/gen_code/gen_test_program)
    """

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        if not data_dir.name == "Generalization":
            data_dir = data_dir / "Generalization"

        gen_file = data_dir / "generalization_dataset.json"
        if not gen_file.exists():
            raise FileNotFoundError(
                f"Generalization dataset: {gen_file}\n"
                f"run experiments/build_generalization_dataset.py"
            )

        print(f"loadGeneralization dataset: {gen_file}")
        with open(gen_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        processed = [self._process_sample(s, model_name) for s in raw]
        self._data = processed[:size] if size else processed
        print(f"Loaded {len(self._data)} testsample")

    def _process_sample(self, sample: dict, model_name: str) -> dict:
        language = sample.get("language", "rust")
        api_name = sample.get("api_name", "")
        api_module = sample.get("api_module", "")
        from_version = sample.get("from_version", "")
        to_version = sample.get("to_version", "")
        signature = sample.get("signature", "")

        edit_query = sample.get("edit_query", "")
        edit_func_sig = sample.get("edit_function_signature", "").strip()
        edit_code = sample.get("edit_code", "")
        edit_test = sample.get("edit_test_program", "")

        gen_query = sample.get("gen_query", "")
        gen_func_sig = sample.get("gen_function_signature", "").strip()
        gen_code = sample.get("gen_code", "")
        gen_test = sample.get("gen_test_program", "")

        if language == "python":
            edit_code = strip_code_comments(edit_code)
            gen_code = strip_code_comments(gen_code)

        if language == "rust":
            api_info = f"{api_module}::{api_name}"
            code_block = f"```rust\n{edit_func_sig}\n```"
        else:
            api_info = f"{api_module}.{api_name}" if "." not in api_name else api_name
            code_block = f"```python\n{edit_func_sig}\n```"

        subject_parts = [
            f"API Information: {api_info}",
            f"API Signature: {signature}",
            f"Task: {edit_query}",
            "Function Signature:",
            code_block,
        ]
        subject_raw = "\n".join(subject_parts)

        is_llama = any(n in model_name for n in ['Llama3', 'Llama-3', 'llama3', 'llama-3'])
        is_qwen = 'Qwen' in model_name or 'qwen' in model_name

        if is_llama:
            subject = get_llama_without_answer(subject_raw)
            answer = edit_code + "<|eot_id|>"
        elif is_qwen:
            subject = get_qwen_without_answer(subject_raw)
            answer = edit_code + "<|im_end|>"
        else:
            subject = subject_raw
            answer = edit_code

        question = subject
        sample_id = sample.get("id", f"{api_name}_{from_version}_to_{to_version}")

        return {
            "id": sample_id,
            "language": language,

            "subject": subject,
            "target": edit_code,
            "question": question,
            "answer": answer,
            "para_question": question,
            "sub_question": [question],
            "category": "code",

            "edit_query": edit_query,
            "edit_function_signature": edit_func_sig,
            "edit_code": edit_code,
            "edit_test_program": edit_test,

            "gen_query": gen_query,
            "gen_function_signature": gen_func_sig,
            "gen_code": gen_code,
            "gen_test_program": gen_test,

            "name": api_name,
            "module": api_module,
            "from_version": from_version,
            "to_version": to_version,
            "signature": signature,
            "source_code": sample.get("source_code", ""),
            "old_source_code": sample.get("old_source_code", ""),
            "documentation": sample.get("documentation", ""),
            "change_type": sample.get("change_type", "signature"),
            "type": sample.get("type", ""),
            "examples": sample.get("examples", []),
            "pair_source": sample.get("pair_source", ""),

            "query": edit_query,
            "function_signature": edit_func_sig,
            "code": edit_code,
            "test_program": edit_test,
        }

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
