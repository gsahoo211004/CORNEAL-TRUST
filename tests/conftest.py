from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def project_root():
    return ROOT


@pytest.fixture
def default_config():
    from src.utils.config import load_config

    return load_config()
