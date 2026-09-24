from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    """Compatibility alias for the legacy tmp fixture used by older tests."""
    return tmp_path
