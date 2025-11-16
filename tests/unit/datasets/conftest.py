#!/usr/bin/env python3
"""
Pytest configuration for dataset unit tests.

Provides fixtures to isolate tests and prevent artifact pollution.

Date: 2025-11-15
Author: Claude Code
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="function", autouse=True)
def isolate_test_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically isolate all tests in this directory to tmp_path.

    This prevents tests from creating persistent artifacts in the working directory.
    Specifically prevents `outputs/test_dataset/` from being created during test runs.

    Applied automatically to all tests in tests/unit/datasets/.
    """
    # Change working directory to tmp_path for isolation
    monkeypatch.chdir(tmp_path)
