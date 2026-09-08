"""Integration test: Module 5's orchestrator running dependency-ordered predictions using
features read from a REAL D1Store (Module 4) — not a hand-built DataFrame.

Uses the same trivial dummy components as the unit tests (no real DT model exists yet — Modules
6-10 are separate, later work). The point here is proving the *wiring*: D1's current-state table
columns feed directly into `DTOrchestrator.run_predictions()` as REQUIRED_FEATURES, and the
resulting per-row predictions correspond correctly to the D1 rows they came from.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.dt_models.d1_model_store import D1Store
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.telemetry.schema import CleanTelemetryRecord, RecordQuality
from tests.dummy_dt_components import DummyAdder, DummyDoubler, DummySummer


def _clean_record(ue_id: str, cell_id: str, sinr_db: float, throughput_mbps: float) -> CleanTelemetryRecord:
    return CleanTelemetryRecord(
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        ue_id=ue_id,
        cell_id=cell_id,
        throughput_mbps=throughput_mbps,
        offered_load_mbps=60.0,
        latency_ms=20.0,
        jitter_ms=3.0,
        packet_loss_pct=1.0,
        prb_utilization_pct=40.0,
        sinr_db=sinr_db,
        rsrp_dbm=-90.0,
        rsrq_db=-9.0,
        ue_count=12,
        ue_speed_mps=3.5,
        source="MOCK",
        quality=RecordQuality(),
    )


def test_orchestrator_runs_predictions_from_real_d1_current_state(tmp_path):
    store = D1Store(
        current_state_path=tmp_path / "current.parquet",
        history_path=tmp_path / "history.parquet",
        quarantine_path=tmp_path / "quarantine.parquet",
        history_retention_rows=1000,
    )
    # Feed D1 directly (bypassing the full telemetry pipeline — Modules 2/3 are tested
    # elsewhere) so this test isolates exactly what it's checking: orchestrator <- D1 wiring.
    store.update_current_state(
        [
            _clean_record("ue-0", "cell-0", sinr_db=10.0, throughput_mbps=50.0),
            _clean_record("ue-1", "cell-0", sinr_db=20.0, throughput_mbps=80.0),
            _clean_record("ue-2", "cell-1", sinr_db=5.0, throughput_mbps=30.0),
        ]
    )

    # Dummy components declare REQUIRED_FEATURES that happen to name real D1 columns
    # ("sinr_db" as "x", conceptually) — here we just reuse D1's own column names directly by
    # renaming the dummy components' feature requirement to match, proving features literally
    # flow from D1 into the orchestrator with no manual reshaping beyond column selection.
    class SinrDoubler(DummyDoubler):
        COMPONENT_NAME = "sinr_doubler"
        REQUIRED_FEATURES = ("sinr_db",)

        def predict(self, features):
            return (features["sinr_db"] * 2).rename(self.OUTPUT_FIELD)

    class ThroughputAdder(DummyAdder):
        COMPONENT_NAME = "throughput_adder"
        DEPENDENCIES = ("sinr_doubler",)
        REQUIRED_FEATURES = ("throughput_mbps",)

        def predict(self, features):
            return (features["throughput_mbps"] + features["a_pred"]).rename(self.OUTPUT_FIELD)

    class CombinedSummer(DummySummer):
        COMPONENT_NAME = "combined_summer"
        DEPENDENCIES = ("sinr_doubler", "throughput_adder")

    registry = DTModelRegistry()
    doubler, adder, summer = SinrDoubler(), ThroughputAdder(), CombinedSummer()
    for component in (doubler, adder, summer):
        component.train(store.get_current_state(), store.get_current_state()["sinr_db"])
        registry.register(component)

    orchestrator = DTOrchestrator(registry)
    current_state = store.get_current_state()  # <-- features read directly from D1
    predictions = orchestrator.run_predictions(current_state)

    assert set(predictions.keys()) == {"sinr_doubler", "throughput_adder", "combined_summer"}
    for name, series in predictions.items():
        assert len(series) == len(current_state) == 3

    # Verify the predictions correspond correctly, row-for-row, to the D1 data they came from.
    for i, (_, row) in enumerate(current_state.iterrows()):
        expected_a = row["sinr_db"] * 2
        expected_b = row["throughput_mbps"] + expected_a
        expected_c = expected_a + expected_b
        assert predictions["sinr_doubler"].iloc[i] == expected_a
        assert predictions["throughput_adder"].iloc[i] == expected_b
        assert predictions["combined_summer"].iloc[i] == expected_c
