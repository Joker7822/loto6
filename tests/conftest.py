"""Pytest configuration for repository-root imports.

GitHub Actions may collect tests with a working import path that does not
include the repository root. This file makes the root-level loto6.py module
importable from tests consistently on local machines and CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
