"""Shared fixtures/helpers for DT prediction-component integration tests (Modules 6+).

`bootstrap_history` runs the REAL telemetry -> preprocessing -> synchronization -> D1 pipeline
(Modules 2-4) once per test session to produce a realistic bootstrap dataset, exactly as
prompt.md §14 describes bootstrap training data arriving. `MockTelemetrySource` is clearly
SOURCE=MOCK (no real NS-3 exporter exists yet — see Module 1 status), but it is not synthetic
noise: its built-in physical correlation model (SINR/PRB/offered-load -> throughput, etc.) gives
a real regressor something genuine to learn — which is exactly what each model's training test
needs to prove. Session-scoped because generating it (1200 records through the real synchronizer)
is the expensive part of these tests and every consumer only reads it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.config import load_settings
from src.dt_models.d1_model_store import D1Store
from src.synchronization.sync import ContinuousSynchronizer
from src.telemetry.mock_source import MockTelemetrySource
from src.telemetry.preprocessing import TelemetryPreprocessor


@pytest.fixture(scope="session")
def bootstrap_history(tmp_path_factory) -> pd.DataFrame:
    tmp_path = tmp_path_factory.mktemp("d1_bootstrap")
    settings = load_settings()

    store = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=settings.storage.history_retention_rows,
    )
    mock_config = settings.telemetry.mock.model_copy(
        update={
            "num_ues": 25,
            "num_cells": 4,
            "missing_field_rate": 0.0,  # bootstrap data should be clean; imputation is Module 2's concern
            "out_of_range_rate": 0.0,
        }
    )
    source = MockTelemetrySource(mock_config, max_records=1200, realtime=False)
    preprocessor = TelemetryPreprocessor(settings.telemetry.preprocessing)
    sync_config = settings.synchronization.model_copy(update={"batch_size": 100})
    synchronizer = ContinuousSynchronizer(source, preprocessor, store, sync_config)

    synchronizer.run()  # bounded source -> blocks until exhausted; real Module 2/3/4 pipeline

    history = store.get_history()
    assert len(history) == 1200  # every generated record made it through validation
    return history


def time_split(history: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/held-out split — avoids any leakage between the two sets."""
    ordered = history.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split_at = int(len(ordered) * train_fraction)
    return ordered.iloc[:split_at], ordered.iloc[split_at:]
