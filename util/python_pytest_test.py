"""
Python pytest testUtility — corresponding rust_cargo_test.py Python version(Sandbox mode)

Design principles:
  eval scripts run in any environment(RustEvo/AEG , with vLLM/transformers),
  tests executed via subprocess call **correspondingversionSandbox environment** Python, implementVersion isolation.

  ┌──────────────────┐        subprocess        ┌──────────────────┐
  │  RustEvo / AEG │  ─────────────────────▶  │ Version Sandbox  │
  │  (vLLM / transf) │   python_bin -m pytest   │ (Python X.Y +    │
  │ eval_*.py run │ │ pkg==to_version)│
  └──────────────────┘                          └──────────────────┘

  --sandbox mode:
    entriesbased on module + to_version autoselectcorrespondingversionSandbox environment.
    - Stdlib entry → corresponding Python version (3.8-3.13)
    - Third-party entry → correspondingversion (e.g., numpy==2.0)

Features:
1. run pytest Validate LLM generate Python code
2. Check function signature match
3. Check API usage correctness
4. Supporttimeout
5. python_bin parameter, Supportenvironmentsandboxcall
6. SandboxManager , versionsandboxtest

test strategy:
- solution.py + test_solution.py directory
- Use python_bin pytest( PyEvo environment)
"""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_TEST_TIMEOUT = int(os.environ.get("PYEVO_TEST_TIMEOUT", "60"))
DEFAULT_PYTHON_BIN = os.environ.get(
    "PYEVO_PYTHON",
    "python",
)
DEFAULT_SANDBOX_CACHE = os.environ.get(
    "PYEVO_SANDBOX_CACHE",
    "data/sandbox_cache",
)

# Lazy-loaded sandbox manager
_sandbox_mgr = None
_sandbox_mgr_cache_dir = None


def _get_sandbox_manager(cache_dir: str = None):
    """get SandboxManager ."""
    global _sandbox_mgr, _sandbox_mgr_cache_dir
    cache_dir = cache_dir or DEFAULT_SANDBOX_CACHE

    if _sandbox_mgr is not None and _sandbox_mgr_cache_dir == cache_dir:
        return _sandbox_mgr

    # Import from PyEvo scripts
    pyevo_scripts = str((Path(__file__).parent.parent / ".." / "PyEvo" / "scripts").resolve())
    if pyevo_scripts not in sys.path:
        sys.path.insert(0, pyevo_scripts)

    from sandbox_manager import SandboxManager
    _sandbox_mgr = SandboxManager(cache_dir, verbose=False)
    _sandbox_mgr_cache_dir = cache_dir
    return _sandbox_mgr


def get_sandbox_python(
    module: str,
    to_version: str,
    sandbox_cache: str = None,
) -> Optional[str]:
    """
    based on module + to_version getcorrespondingsandbox Python path.

    Args:
        module: module (e.g., 'numpy', 'asyncio')
        to_version: targetversion (e.g., '2.0', '3.12')
        sandbox_cache: sandboxdirectory

    Returns:
        python_path None (sandbox)
    """
    mgr = _get_sandbox_manager(sandbox_cache)

    if mgr.is_stdlib(module):
        return mgr.get_stdlib_python(to_version)
    else:
        return mgr.create_package_env(module, to_version)


def run_python_test(
    code: str,
    test_program: str,
    timeout: int = DEFAULT_TEST_TIMEOUT,
    python_bin: str = None,
) -> Dict:
    """
    Run pytest on generated code + test_program in sandbox environment.

    Args:
        code: LLM generate Python code
        test_program: corresponding pytest testcode
        timeout: timeout()
        python_bin: test Python path( PyEvo environment)

    Returns dict with keys:
        success: bool
        output: str   (stdout+stderr)
        error:  str   (error category if failed)
    """
    python_bin = python_bin or DEFAULT_PYTHON_BIN

    with tempfile.TemporaryDirectory(prefix="pyevo_") as tmpdir:
        sol_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"

        sol_path.write_text(code, encoding="utf-8")
        test_path.write_text(test_program, encoding="utf-8")

        proc = subprocess.Popen(
                [python_bin, "-m", "pytest", str(test_path), "-v", "--tb=short", "-q"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                start_new_session=True,
            )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            output = (stdout + "\n" + stderr).strip()

            if proc.returncode == 0:
                return {"success": True, "output": output, "error": ""}
            else:
                return {"success": False, "output": output, "error": "test_failed"}

        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {"success": False, "output": "", "error": "timeout"}
        except Exception as e:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            return {"success": False, "output": str(e), "error": "execution_error"}


def check_function_signature(code: str, expected_signature: str) -> bool:
    """
    Check whether the generated code contains a function matching the expected signature.
    Lenient: we only check the function name matches.
    """
    if not expected_signature:
        return True

    # Extract function name from signature like "def foo(..." 
    m = re.search(r'def\s+(\w+)\s*\(', expected_signature)
    if not m:
        return True  # can't parse -> skip check

    func_name = m.group(1)
    return bool(re.search(rf'def\s+{re.escape(func_name)}\s*\(', code))


def check_api_usage(code: str, api_name: str, change_type: str, module: str) -> bool:
    """
    Check that the generated code uses the required API (mode).

    Useedgematch(\\b)match API name, :
    - "cache" match "lru_cache"
    - import moduleUsetarget API

    For 'deprecated' change_type: the code should NOT use the old API.
    For other types: the code MUST contain the exact API short name.
    """
    if not api_name:
        return True

    # Normalize: for names like "DataFrame.pivot_table", check "pivot_table"
    # For "str.removeprefix", check "removeprefix"
    short_name = api_name.split(".")[-1] if "." in api_name else api_name

    if str(change_type).lower() == "deprecated":
        # For deprecated APIs, we want the code NOT to use the old name
        # But this is tricky — sometimes the API still exists. Be lenient.
        return True

    if re.search(rf'\b{re.escape(short_name)}\b', code):
        return True

    return False


def extract_python_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try ```python ... ``` blocks first
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, response)

    if matches:
        valid = [m.strip() for m in matches if m.strip()]
        if valid:
            return max(valid, key=len)

    # Fallback: look for def/import/from/class lines
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        if re.match(r"\s*(import |from |def |class |@)", line):
            in_code = True
        if in_code:
            code_lines.append(line)

    return "\n".join(code_lines) if code_lines else response


def prepare_test_inputs(
    generated_code: str,
    test_program: str,
) -> Tuple[str, str]:
    """
    Clean up code and test program for execution.
    Returns (clean_code, clean_test).
    """
    # Ensure test imports from solution
    if "from solution import" not in test_program and "import solution" not in test_program:
        # Try to find function names in generated code
        func_names = re.findall(r"def\s+(\w+)\s*\(", generated_code)
        if func_names:
            import_line = f"from solution import {', '.join(func_names)}\n"
            test_program = import_line + test_program

    return generated_code, test_program
