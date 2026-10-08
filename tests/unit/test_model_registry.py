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
from src.data.oai_dataset import ALL_FEATURES
from src.dt_models.dl_bler import DlBlerModel
from src.registry.model_registry import ModelRegistry, ModelRegistryError

SETTINGS = load_settings()


def _oai_features(n: int) -> pd.DataFrame:
    return pd.DataFrame({col: np.random.uniform(0, 1, n) for col in ALL_FEATURES})


def _trained_model() -> DlBlerModel:
    model = DlBlerModel.from_settings(SETTINGS)
    n = 40
    X = _oai_features(n)
    y = pd.Series(np.random.uniform(0, 1, n), name="dl_bler")
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

    assert version.component == "dl_bler"
    assert version.version_id == "dl_bler-v1"
    assert version.model_class == "DlBlerModel"
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
    assert v1.version_id == "dl_bler-v1"
    assert v2.version_id == "dl_bler-v2"
    assert v2.parent_version_id == v1.version_id


def test_get_current_version_returns_only_production_status(tmp_path):
    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    candidate = registry.register_version(
        component_instance=_trained_model(), adaptation_type="recalibrate", parent_version_id="dl_bler-v1",
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )
    current = registry.get_current_version("dl_bler")
    assert current.version_id == "dl_bler-v1"
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

    promoted = registry.promote("dl_bler", v2.version_id)

    assert promoted.status == "production"
    assert registry.get_version("dl_bler", v1.version_id).status == "superseded"
    assert registry.get_current_version("dl_bler").version_id == v2.version_id


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
    registry.promote("dl_bler", v2.version_id)

    production_versions = [v for v in registry.list_versions("dl_bler") if v.status == "production"]
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

    rejected = registry.reject("dl_bler", v2.version_id)

    assert rejected.status == "rejected"
    assert registry.get_current_version("dl_bler").version_id == v1.version_id  # untouched


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
    registry.promote("dl_bler", v2.version_id)

    rolled_back = registry.rollback("dl_bler")

    assert rolled_back.version_id == v1.version_id
    assert rolled_back.status == "production"
    assert registry.get_version("dl_bler", v2.version_id).status == "superseded"


def test_rollback_raises_when_nothing_to_roll_back_to(tmp_path):
    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    with pytest.raises(ModelRegistryError):
        registry.rollback("dl_bler")


def test_promote_raises_for_unknown_version(tmp_path):
    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    with pytest.raises(ModelRegistryError):
        registry.promote("dl_bler", "dl_bler-v999")


def test_get_version_raises_for_unknown_version(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ModelRegistryError):
        registry.get_version("dl_bler", "dl_bler-v1")


def test_load_artifact_into_restores_a_working_component(tmp_path):
    registry = _registry(tmp_path)
    model = _trained_model()
    version = registry.register_version(
        component_instance=model, adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )

    fresh = DlBlerModel.from_settings(SETTINGS)
    assert not fresh.is_trained
    registry.load_artifact_into(fresh, version)

    assert fresh.is_trained
    X = _oai_features(1)
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

    assert reloaded.get_current_version("dl_bler").version_id == v1.version_id
    assert reloaded.get_version("dl_bler", v1.version_id).evaluation_metrics["rmse"] == pytest.approx(1.23)
    assert len(reloaded.list_versions("dl_bler")) == 1


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
    registry.promote("dl_bler", v2.version_id)

    reloaded = _registry(tmp_path)

    assert reloaded.get_current_version("dl_bler").version_id == v2.version_id
    assert reloaded.get_version("dl_bler", v1.version_id).status == "superseded"


def test_versions_are_independent_across_components(tmp_path):
    from src.dt_models.ul_bler import UlBlerModel

    registry = _registry(tmp_path)
    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    ul_bler_model = UlBlerModel.from_settings(SETTINGS)
    n = 30
    X = _oai_features(n)
    y = pd.Series(np.random.uniform(0, 5, n), name="ul_bler")
    ul_bler_model.train(X, y)
    registry.register_version(
        component_instance=ul_bler_model, adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )

    assert registry.get_current_version("dl_bler").version_id == "dl_bler-v1"
    assert registry.get_current_version("ul_bler").version_id == "ul_bler-v1"
    assert len(registry.list_versions("dl_bler")) == 1
    assert len(registry.list_versions("ul_bler")) == 1


# --- source_path() / list_component_names() — used by ContinuousOrchestrator's hot-swap path -----


def test_source_path_is_none_for_a_recalibration_bootstrap_version(tmp_path):
    registry = _registry(tmp_path)
    version = registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    assert registry.source_path(version) is None


def test_source_path_resolves_to_the_real_saved_source_file(tmp_path):
    registry = _registry(tmp_path)
    artifact = tmp_path / "candidate.joblib"
    artifact.write_bytes(b"not a real joblib file, just bytes for this path-resolution test")
    source = tmp_path / "candidate.py"
    source.write_text("class Rebuilt: pass\n")

    version = registry.register_version_from_artifact(
        component="dl_bler", model_class="Rebuilt", artifact_source_path=artifact, source_code_path=source,
        dependencies=(), feature_schema=(), output_field="dl_bler_pred", adaptation_type="regenerate",
        parent_version_id="dl_bler-v1", training_window={}, evaluation_window={}, evaluation_metrics={},
    )

    resolved = registry.source_path(version)
    assert resolved is not None
    assert resolved.exists()
    assert resolved.read_text() == "class Rebuilt: pass\n"


def test_list_component_names_returns_every_tracked_component(tmp_path):
    from src.dt_models.ul_bler import UlBlerModel

    registry = _registry(tmp_path)
    assert registry.list_component_names() == []

    registry.register_version(
        component_instance=_trained_model(), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
    )
    registry.register_version(
        component_instance=UlBlerModel.from_settings(SETTINGS), adaptation_type="bootstrap", parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="candidate",
    )
    assert set(registry.list_component_names()) == {"dl_bler", "ul_bler"}
