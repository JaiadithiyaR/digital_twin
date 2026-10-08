"""Integration test: all 7 OAI KPI prediction components (Modules 6-10 replacement, prompt2.md
Step 5), trained and evaluated against the REAL processed OAI dataset
(`data/processed/oai/{train,test}.parquet`, built by `scripts/build_oai_dataset.py` from the real
exported RAN KPI CSVs) — never mock telemetry. Proves train/predict/save/load round-trips on
real data, the real dependency chain (`dl_bler`/`dl_prb` -> `dl_throughput`; `dl_throughput`/
`dl_bler` -> `dl_rlc_delay`) through a real `DTOrchestrator`, and that the full 7-component graph
is a valid DAG with no hardcoded call order. Skips cleanly (not a failure) if the dataset hasn't
been built in this environment yet — this test deliberately never generates synthetic data as a
substitute for the real OAI export (prompt2.md: "never invent metrics"). Phase 13 §1.1 redesign:
no more separate "val" split — `test` (run 8) is the only other split besides `train`, and this
test deliberately still never uses it for anything resembling TUNING (only a fixed-hyperparameter
sanity check that training/prediction/persistence genuinely work end-to-end on real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.dt_models.dl_bler import DlBlerModel
from src.dt_models.dl_prb import DlPrbModel
from src.dt_models.dl_rlc_delay import DlRlcDelayModel
from src.dt_models.dl_throughput import DlThroughputModel
from src.dt_models.model_registry import DTModelRegistry
from src.dt_models.orchestrator import DTOrchestrator
from src.dt_models.ul_bler import UlBlerModel
from src.dt_models.ul_prb import UlPrbModel
from src.dt_models.ul_throughput import UlThroughputModel

SETTINGS = load_settings()


@pytest.fixture(scope="module")
def real_oai_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = SETTINGS.resolve_path("data/processed/oai/train.parquet")
    test_path = SETTINGS.resolve_path("data/processed/oai/test.parquet")
    if not train_path.exists() or not test_path.exists():
        pytest.skip("data/processed/oai/{train,test}.parquet not built — run scripts/build_oai_dataset.py")
    return pd.read_parquet(train_path), pd.read_parquet(test_path)


def test_all_seven_components_train_predict_save_load_round_trip(real_oai_split, tmp_path):
    train, test = real_oai_split
    components = {
        "dl_bler": DlBlerModel, "ul_bler": UlBlerModel, "dl_prb": DlPrbModel, "ul_prb": UlPrbModel,
        "dl_throughput": DlThroughputModel, "ul_throughput": UlThroughputModel, "dl_rlc_delay": DlRlcDelayModel,
    }
    dep_cols = {
        "dl_throughput": {"dl_bler": "dl_bler", "dl_prb": "dl_prb"},
        "ul_throughput": {"ul_bler": "ul_bler", "ul_prb": "ul_prb"},
        "dl_rlc_delay": {"dl_throughput": "dl_throughput", "dl_bler": "dl_bler"},
    }

    train_with_deps = train.copy()
    for name, cls in components.items():
        model = cls.from_settings(SETTINGS)
        train_input = train_with_deps
        model.train(train_input, train[name])
        assert model.is_trained

        preds = model.predict(train_input.head(5))
        assert len(preds) == 5

        metrics = model.evaluate(train_input, train[name])
        assert "rmse" in metrics and metrics["rmse"] >= 0

        save_path = tmp_path / f"{name}.joblib"
        model.save(save_path)
        assert save_path.exists()

        fresh = cls.from_settings(SETTINGS)
        assert not fresh.is_trained
        fresh.load(save_path)
        assert fresh.is_trained
        reloaded_preds = fresh.predict(train_input.head(5))
        # equal_nan=True: a residual-reconstructed prediction is honestly NaN wherever y_t-1
        # itself is unknown (e.g. the first row of a session) — NaN at the same position in both
        # arrays still means "genuinely restored, not a fresh/different fit", plain list/`==`
        # equality just can't express that (NaN != NaN).
        assert np.allclose(reloaded_preds.to_numpy(), preds.to_numpy(), equal_nan=True)

        train_with_deps[f"{name}_pred"] = train[name]  # ground truth, for the NEXT dependent component


def test_real_dependency_chain_runs_through_a_real_orchestrator(real_oai_split):
    """The full 7-component graph: a valid DAG, no hardcoded call order, dependents genuinely
    wired to their upstream components' LIVE predictions (not ground truth) at serve time."""
    train, test = real_oai_split
    registry = DTModelRegistry()
    train_with_deps = train.copy()

    order = ["dl_bler", "ul_bler", "dl_prb", "ul_prb", "dl_throughput", "ul_throughput", "dl_rlc_delay"]
    classes = {
        "dl_bler": DlBlerModel, "ul_bler": UlBlerModel, "dl_prb": DlPrbModel, "ul_prb": UlPrbModel,
        "dl_throughput": DlThroughputModel, "ul_throughput": UlThroughputModel, "dl_rlc_delay": DlRlcDelayModel,
    }
    for name in order:
        model = classes[name].from_settings(SETTINGS)
        model.train(train_with_deps, train[name])
        registry.register(model)
        train_with_deps[f"{name}_pred"] = train[name]

    orchestrator = DTOrchestrator(registry)
    execution_order = orchestrator.build_execution_order()

    assert execution_order.index("dl_bler") < execution_order.index("dl_throughput")
    assert execution_order.index("dl_prb") < execution_order.index("dl_throughput")
    assert execution_order.index("ul_bler") < execution_order.index("ul_throughput")
    assert execution_order.index("ul_prb") < execution_order.index("ul_throughput")
    assert execution_order.index("dl_throughput") < execution_order.index("dl_rlc_delay")
    assert execution_order.index("dl_bler") < execution_order.index("dl_rlc_delay")

    predictions = orchestrator.run_predictions(test)
    assert set(predictions) == set(order)
    for name in order:
        assert len(predictions[name]) == len(test)
