"""Shared test helpers."""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_execution  # noqa: E402
from execution import (  # noqa: E402
    broker, gates, market_data, orders, risk, safeguards, sizing, trading,
)

MODULES = (run_execution, broker, gates, market_data, orders, risk, safeguards, sizing, trading)


def patch_name(testcase, name, **mock_kwargs):
    """Replace `name` with ONE shared mock in every module that imported it.

    The functions under test look names up in whichever module they live in, so
    patching only one module would silently miss the others. The patches are
    undone automatically when the test finishes. Returns the shared mock.
    """
    shared = mock.Mock(**mock_kwargs)
    patched = 0
    for module in MODULES:
        if hasattr(module, name):
            patcher = mock.patch.object(module, name, shared)
            patcher.start()
            testcase.addCleanup(patcher.stop)
            patched += 1
    if not patched:
        raise AttributeError(f"{name} does not exist in any execution module")
    return shared
