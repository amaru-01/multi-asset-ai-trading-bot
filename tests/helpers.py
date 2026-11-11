"""Shared test helpers."""

import importlib
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import retrain  # noqa: E402
import run_execution  # noqa: E402
from execution import (  # noqa: E402
    broker, gates, market_data, orders, risk, safeguards, sizing, trading,
)

MODULES = [run_execution, broker, gates, market_data, orders, risk, safeguards, sizing, trading]

# The retraining code is split across the training package (and retrain.py re-exports
# it). Include whichever of these modules exist.
RETRAIN_MODULES = [retrain]
for _name in (
    "settings", "data", "gates", "threshold", "candidate", "incumbent", "status", "report",
):
    try:
        RETRAIN_MODULES.append(importlib.import_module(f"training.{_name}"))
    except ImportError:
        pass


def patch_name(testcase, name, modules=None, **mock_kwargs):
    """Replace `name` with ONE shared mock in every module that imported it.

    The functions under test look names up in whichever module they live in, so
    patching only one module would silently miss the others. The patches are
    undone automatically when the test finishes. Returns the shared mock.
    """
    shared = mock.Mock(**mock_kwargs)
    patched = 0
    for module in modules or MODULES:
        if hasattr(module, name):
            patcher = mock.patch.object(module, name, shared)
            patcher.start()
            testcase.addCleanup(patcher.stop)
            patched += 1
    if not patched:
        raise AttributeError(f"{name} does not exist in any searched module")
    return shared


def patch_value(testcase, name, value, modules=None):
    """Like patch_name but substitutes a plain value (a constant or path)."""
    patched = 0
    for module in modules or MODULES:
        if hasattr(module, name):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            testcase.addCleanup(patcher.stop)
            patched += 1
    if not patched:
        raise AttributeError(f"{name} does not exist in any searched module")
    return value
