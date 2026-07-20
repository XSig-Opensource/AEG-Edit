import io
import json
import tokenize
from pathlib import Path


def strip_code_comments(code: str) -> str:
    """
    Strip comments and docstrings from Python code using tokenize + ast.
    Keeps code logic to reduce answer token count for model editing.
    """
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
            pass  # Continue anyway, just skip docstrings

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
    """Strip trailing # comments from a line, correctly handling # inside strings."""
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


class PyEvoDataset:
    """
    PyEvo Dataset loader for Python API evolution knowledge editing tasks.
    """

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        data_dir = Path(data_dir)

        if not data_dir.name == "PyEvo":
            data_dir = data_dir / "PyEvo"

        pyevo_file = data_dir / "PyEvo.json"
        if not pyevo_file.exists():
            raise FileNotFoundError(
                f"PyEvo data file not found: {pyevo_file}\n"
                f"Please ensure the file exists in {data_dir}"
            )

        print(f"Loading PyEvo dataset: {pyevo_file}")
        with open(pyevo_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        processed_data = [self._process_sample(s, model_name) for s in raw]
        self._data = processed_data[:size] if size else processed_data
        print(f"Loaded {len(self._data)} samples")

    def _process_sample(self, sample: dict, model_name: str) -> dict:
        api_name     = sample.get("name", "")
        api_module   = sample.get("module", "")
        from_version = sample.get("from_version", "unknown")
        to_version   = sample.get("to_version", "unknown")

        query          = sample.get("rephrased_query") or sample.get("query", "")
        original_query = sample.get("query", "")
        signature      = sample.get("signature", "")
        func_sig       = sample.get("function_signature", "").strip()
        raw_code       = sample.get("code", "") or ""
        clean_code     = strip_code_comments(raw_code)

        if api_name and api_module and query:
            api_info = f"{api_module}.{api_name}" if "." not in api_name else api_name
            subject_parts = [
                f"API Information: {api_info}",
                f"API Signature: {signature}",
                f"Task: {query}",
                "Function Signature:",
                f"```python\n{func_sig}\n```",
            ]
            subject_raw = "\n".join(subject_parts)
        else:
            subject_raw = query or f"Implement using {api_module}.{api_name}"

        if any(n in model_name for n in ["Llama3", "Llama-3", "llama3", "llama-3"]):
            subject = get_llama_without_answer(subject_raw)
            answer  = clean_code + "<|eot_id|>"
        elif "Qwen" in model_name or "qwen" in model_name:
            subject = get_qwen_without_answer(subject_raw)
            answer  = clean_code + "<|im_end|>"
        else:
            subject = subject_raw
            answer  = clean_code

        question   = subject
        sample_id  = f"{api_name}_{from_version}_to_{to_version}" if api_name else "unknown"

        return {
            "id":              sample_id,
            "subject":         subject,
            "target":          clean_code,
            "question":        question,
            "answer":          answer,
            "para_question":   question,
            "sub_question":    [question],
            "test_program":    sample.get("test_program", ""),
            "category":        "code",
            "module":          api_module,
            "from_version":    from_version,
            "to_version":      to_version,
            "query":           original_query,
            "rephrased_query": sample.get("rephrased_query", ""),
            "function_signature": func_sig,
            "name":            api_name,
            "source_code":     sample.get("source_code", ""),
            "old_source_code": sample.get("old_source_code", ""),
            "signature":       signature,
            "old_signature":   sample.get("old_signature", ""),
            "documentation":   sample.get("documentation", ""),
            "change_type":     sample.get("change_type", "signature"),
            "type":            sample.get("type", "function"),
            "examples":        sample.get("examples", []),
            "description":     sample.get("description", ""),
            "code":            clean_code,
        }

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
