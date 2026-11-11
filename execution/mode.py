"""Paper/live mode helpers (their own module to avoid circular imports)."""

import os


def is_paper_mode() -> bool:
    """Paper trading unless ALPACA_PAPER is explicitly set to "false"."""
    return os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"


def mode_label() -> str:
    return "PAPER" if is_paper_mode() else "LIVE"
