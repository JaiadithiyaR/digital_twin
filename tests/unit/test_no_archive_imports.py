"""Permanent guard (prompt2.md Step 2): no module under `src/` may import from `archive/`
(the retired NS-3/5G-LENA integration) or import `zmq` (the retired ZeroMQ telemetry transport).

This is a static source-text check deliberately independent of whether the modules in question
are importable at all (archive/ns3_sim/zmq_source.py may not even be importable once `pyzmq` is
removed from requirements.txt) — it must keep working even if `archive/` is deleted entirely or
`zmq` is uninstalled, so it reads every `.py` file's text directly rather than trying to import it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _all_src_py_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _imported_module_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _all_src_py_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_module_under_src_imports_archive_or_zmq(path: Path) -> None:
    roots = _imported_module_roots(path)
    assert "archive" not in roots, f"{path} imports from the retired archive/ tree"
    assert "zmq" not in roots, f"{path} imports the retired zmq transport"


def test_this_guard_actually_found_source_files() -> None:
    """Sanity check on the test itself: if `_all_src_py_files()` ever returned an empty list
    (e.g. a path typo), every parametrized case above would vacuously "pass" without checking
    anything — this fails loudly instead."""
    assert len(_all_src_py_files()) > 10
