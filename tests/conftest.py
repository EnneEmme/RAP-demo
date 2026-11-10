"""Shared pytest configuration ensuring run cache outputs go to temporary directories."""

from __future__ import annotations

from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def _isolate_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect RAP_RUNS_DIR to an isolated temporary directory for every test."""
    monkeypatch.setenv("RAP_RUNS_DIR", str(tmp_path / "runs"))
