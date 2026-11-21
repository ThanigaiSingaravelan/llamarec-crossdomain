"""Smoke test: every project module imports cleanly.

Catches missing imports / undefined names that only surface at runtime.
A module is skipped (not failed) only when a third-party dependency is not
installed locally; in CI all of requirements.txt is installed, so all must pass.
"""
import importlib
import os

import pytest

PROJECT_PACKAGES = {"core", "evaluation", "preprocessing", "webapp"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _discover_modules():
    mods = []
    for pkg in sorted(PROJECT_PACKAGES):
        pkg_dir = os.path.join(ROOT, pkg)
        for name in sorted(os.listdir(pkg_dir)):
            if name.endswith(".py") and name != "__init__.py":
                mods.append(f"{pkg}.{name[:-3]}")
    return mods


@pytest.mark.parametrize("module", _discover_modules())
def test_module_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        top = (exc.name or "").split(".")[0]
        if top in PROJECT_PACKAGES:
            raise
        if os.environ.get("CI"):
            raise
        pytest.skip(f"optional dependency not installed locally: {exc.name}")
