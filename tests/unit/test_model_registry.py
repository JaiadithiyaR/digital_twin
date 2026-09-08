"""Unit tests for ModelRegistry (versioned artifact storage, prompt.md §46/§29).

Needed by Module 14 (this turn) and, unchanged, by Modules 15/16 later — every adaptation agent
registers its candidates the same way. Covers: version creation + artifact persistence, the
current/previous-version distinction, promote/reject/rollback, and — critically — that the index
survives a process restart (a fresh `ModelRegistry` instance pointed at the same paths sees
exactly what a prior instance wrote), mirroring `D1Store`'s own persistence test discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.dt_models.throughput import ThroughputModel
from src.registry.model_registry import ModelRegistry, ModelRegistryError

SETTINGS = load_settings()


def _trained_model() -> ThroughputModel:
    model = ThroughputModel.from_settings(SETTINGS)
    n = 40
    X = pd.DataFrame(
        {
            "offered_load_mbps": np.random.uniform(1, 60, n),
            "prb_utilization_pct": np.random.uniform(1, 100, n),
            "sinr_db": np.random.uniform(-5, 30, n),
            "rsrp_dbm": np.random.uniform(-120, -60, n),
            "rsrq_db": np.random.uniform(-15, -3, n),
            "ue_count": np.random.randint(1, 10, n),
            "ue_speed_mps": np.random.uniform(0, 15, n),
        }
    )
    y = pd.Series(np.random.uniform(1, 50, n), name="throughput_mbps")
    model.train(X, y)
    return model


def _registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")


def test_register_version_persists_artifact_and_returns_metadata(tmp_path):
    registry = _registry(tmp_path)
    model = _trained_model()

    version = registry.register_version(
        component_instance=model,
        adaptation_type="bootstrap",
        parent_version_id=None,
        training_window={"n_rows": 40},
        evaluation_window={"n_rows": 0},
        evaluation_metrics={"rmse": 1.0, "mae": 0.8},
        status="production",
    )

    assert version.component == "throughput"
    assert version.version_id == "throughput-v1"
    assert version.model_class == "ThroughputModel"
    assert version.dependencies == ()
    assert version.status == "production"
    artifact_path = tmp_path / "models" / version.artifact_path
    assert artifact_path.exists()


def test_version_ids_increment_per_component(tmp_path):
    registry = _registry(tmp_path)
    v1 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    v2 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id=v1.version_id,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )
    assert v1.version_id == "throughput-v1"
    assert v2.version_id == "throughput-v2"
    assert v2.parent_version_id == v1.version_id


def test_get_current_version_returns_only_production_status(tmp_path):
    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    candidate = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id="throughput-v1",
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )
    current = registry.get_current_version("throughput")
    assert current.version_id == "throughput-v1"
    assert current.version_id != candidate.version_id


def test_get_current_version_none_when_component_unknown(tmp_path):
    registry = _registry(tmp_path)
    assert registry.get_current_version("unknown_component") is None


def test_promote_makes_target_production_and_supersedes_previous(tmp_path):
    registry = _registry(tmp_path)
    v1 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    v2 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id=v1.version_id,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )

    promoted = registry.promote("throughput", v2.version_id)

    assert promoted.status == "production"
    assert registry.get_version("throughput", v1.version_id).status == "superseded"
    assert registry.get_current_version("throughput").version_id == v2.version_id


def test_only_one_production_version_at_a_time(tmp_path):
    registry = _registry(tmp_path)
    v1 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    v2 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id=v1.version_id,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )
    registry.promote("throughput", v2.version_id)

    production_versions = [v for v in registry.list_versions("throughput") if v.status == "production"]
    assert len(production_versions) == 1


def test_reject_marks_candidate_rejected_never_touches_production(tmp_path):
    registry = _registry(tmp_path)
    v1 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    v2 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id=v1.version_id,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )

    rejected = registry.reject("throughput", v2.version_id)

    assert rejected.status == "rejected"
    assert registry.get_current_version("throughput").version_id == v1.version_id  # untouched


def test_rollback_reverts_to_most_recently_superseded_version(tmp_path):
    registry = _registry(tmp_path)
    v1 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    v2 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id=v1.version_id,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )
    registry.promote("throughput", v2.version_id)

    rolled_back = registry.rollback("throughput")

    assert rolled_back.version_id == v1.version_id
    assert rolled_back.status == "production"
    assert registry.get_version("throughput", v2.version_id).status == "superseded"


def test_rollback_raises_when_nothing_to_roll_back_to(tmp_path):
    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    with pytest.raises(ModelRegistryError):
        registry.rollback("throughput")


def test_promote_raises_for_unknown_version(tmp_path):
    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    with pytest.raises(ModelRegistryError):
        registry.promote("throughput", "throughput-v999")


def test_get_version_raises_for_unknown_version(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ModelRegistryError):
        registry.get_version("throughput", "throughput-v1")


def test_load_artifact_into_restores_a_working_component(tmp_path):
    registry = _registry(tmp_path)
    model = _trained_model()
    version = registry.register_version(
        component_instance=model, adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )

    fresh = ThroughputModel.from_settings(SETTINGS)
    assert not fresh.is_trained
    registry.load_artifact_into(fresh, version)

    assert fresh.is_trained
    X = pd.DataFrame(
        {
            "offered_load_mbps": [10.0],
            "prb_utilization_pct": [50.0],
            "sinr_db": [15.0],
            "rsrp_dbm": [-90.0],
            "rsrq_db": [-9.0],
            "ue_count": [5],
            "ue_speed_mps": [3.0],
        }
    )
    preds = fresh.predict(X)
    assert len(preds) == 1


def test_registry_persists_across_process_restart(tmp_path):
    registry = _registry(tmp_path)
    v1 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={"n_rows": 40}, evaluation_window={}, evaluation_metrics={"rmse": 1.23},
        status="production",
    )

    reloaded = _registry(tmp_path)  # a fresh instance, same paths — simulates a process restart

    assert reloaded.get_current_version("throughput").version_id == v1.version_id
    assert reloaded.get_version("throughput", v1.version_id).evaluation_metrics["rmse"] == pytest.approx(1.23)
    assert len(reloaded.list_versions("throughput")) == 1


def test_promotion_persists_across_reload(tmp_path):
    registry = _registry(tmp_path)
    v1 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    v2 = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id=v1.version_id,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )
    registry.promote("throughput", v2.version_id)

    reloaded = _registry(tmp_path)

    assert reloaded.get_current_version("throughput").version_id == v2.version_id
    assert reloaded.get_version("throughput", v1.version_id).status == "superseded"


def test_versions_are_independent_across_components(tmp_path):
    from src.dt_models.packet_loss import PacketLossModel

    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    packet_loss_model = PacketLossModel.from_settings(SETTINGS)
    n = 30
    X = pd.DataFrame(
        {
            "sinr_db": np.random.uniform(-5, 30, n),
            "rsrp_dbm": np.random.uniform(-120, -60, n),
            "rsrq_db": np.random.uniform(-15, -3, n),
            "prb_utilization_pct": np.random.uniform(1, 100, n),
            "ue_count": np.random.randint(1, 10, n),
            "offered_load_mbps": np.random.uniform(1, 60, n),
        }
    )
    y = pd.Series(np.random.uniform(0, 5, n), name="packet_loss_pct")
    packet_loss_model.train(X, y)
    registry.register_version(
        component_instance=packet_loss_model, adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )

    assert registry.get_current_version("throughput").version_id == "throughput-v1"
    assert registry.get_current_version("packet_loss").version_id == "packet_loss-v1"
    assert len(registry.list_versions("throughput")) == 1
    assert len(registry.list_versions("packet_loss")) == 1
