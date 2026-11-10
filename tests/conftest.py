"""Shared pytest configuration ensuring run cache and model config stay isolated per test."""

from __future__ import annotations

from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def _isolate_runs_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect RAP_RUNS_DIR and RAP_MODELS_CONFIG to isolated temp paths for every test."""
    monkeypatch.setenv("RAP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RAP_MODELS_CONFIG", str(tmp_path / "models.json"))
