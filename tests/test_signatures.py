"""
Guard test: verify that all keyword arguments passed at call sites
actually exist in the function signatures they target.

This prevents the recurring bug where app/main.py passes a kwarg
(e.g. expected_move=) that doesn't exist in the scoring function.
"""

import ast
import inspect
import os

import pytest

from scoring.covered_call import score_covered_calls
from scoring.cash_secured_put import score_cash_secured_puts
from scoring.engine import run_screener


# ---------------------------------------------------------------------------
# Helper: extract kwargs from all call sites in the project
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

_WATCHED_FUNCTIONS = {
    "score_covered_calls": score_covered_calls,
    "score_cash_secured_puts": score_cash_secured_puts,
    "run_screener": run_screener,
}


def _collect_call_sites():
    """
    Walk every .py file in the project and yield
    (file, line, func_name, [kwarg_names]) for each call to a watched function.
    """
    for root, dirs, files in os.walk(_PROJECT_ROOT):
        # Skip venv and cache dirs
        dirs[:] = [d for d in dirs if d not in (".venv", "__pycache__", ".git")]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    tree = ast.parse(fh.read(), filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                # Resolve function name (simple Name or attribute access)
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name not in _WATCHED_FUNCTIONS:
                    continue
                kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]
                if kwargs:
                    yield (path, node.lineno, func_name, kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignatureMatch:
    """Every kwarg at every call site must exist in the target function."""

    @pytest.mark.parametrize(
        "path,lineno,func_name,kwargs",
        list(_collect_call_sites()),
        ids=[
            f"{os.path.basename(p)}:{ln}:{fn}"
            for p, ln, fn, _ in _collect_call_sites()
        ],
    )
    def test_kwargs_exist_in_signature(self, path, lineno, func_name, kwargs):
        func = _WATCHED_FUNCTIONS[func_name]
        valid_params = set(inspect.signature(func).parameters.keys())
        for kw in kwargs:
            assert kw in valid_params, (
                f"{path}:{lineno} — {func_name}() called with "
                f"kwarg '{kw}' which is not in its signature.\n"
                f"  Valid params: {sorted(valid_params)}"
            )

    def test_at_least_one_call_site_found(self):
        """Sanity: make sure the AST walker actually finds call sites."""
        sites = list(_collect_call_sites())
        assert len(sites) >= 2, (
            "Expected to find call sites for watched functions. "
            "Check that app/main.py and tests still call them."
        )
