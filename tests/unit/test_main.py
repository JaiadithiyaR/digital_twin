"""Unit tests for `src/main.py`'s pure integration-glue logic — per-component previous-outcome
tracking (feeding the Decision & Root-Cause Analysis Agent's context, prompt.md §18) and the
DT-model bootstrap dispatch (load-existing vs. train-new). Fast, no real telemetry pipeline or
model training involved except where explicitly noted. The REQUIRED real, end-to-end "run one full
cycle, confirm telemetry never stopped" proof is `scripts/run_orchestrator_demo.py` (a real,
permanent, repo-tracked validation script — see CLAUDE.md's Phase 11 entry for the actual observed
numbers from a real run) plus the lighter-weight, still-genuinely-real
`tests/integration/test_main_orchestrator.py`. The decision-CONTEXT-building logic itself
(fidelity vector, unified score, network-state summary, RAG retrieval) is `src.adaptation.
decision_context`'s own concern, tested in `tests/unit/test_decision_context.py` — not duplicated
here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.adaptation.decision_context import PreviousOutcome
from src.common.config import Secrets, load_settings
from src.dt_models.dl_bler import DlBlerModel
from src.dt_models.model_registry import DTModelRegistry
from src.main import _DT_COMPONENT_CLASSES, _OUTPUT_FIELDS, _TARGET_COLUMNS, ContinuousOrchestrator
from src.registry.model_registry import ModelRegistry

SETTINGS = load_settings()
NO_KEY_SECRETS = Secrets(google_api_key=None)

# A hand-written stand-in for an LLM-generated regenerate/expand_scope candidate — a distinctive,
# constant prediction (999.0) so a test can prove a live prediction genuinely came from THIS
# dynamically-loaded class, never the original statically-imported one.
_DYNAMIC_SOURCE = '''
import pandas as pd
from src.dt_models.base import DTComponent

class RebuiltDlBler(DTComponent):
    COMPONENT_NAME = "dl_bler"
    DEPENDENCIES = ()
    REQUIRED_FEATURES = ("pusch_snr_avg_lag1",)
    OUTPUT_FIELD = "dl_bler_pred"

    def __init__(self):
        self._trained = False

    @property
    def is_trained(self):
        return self._trained

    def train(self, features, targets):
        self._trained = True

    def predict(self, features):
        return pd.Series([999.0] * len(features), index=features.index, name=self.OUTPUT_FIELD)

    def evaluate(self, features, targets):
        return {"rmse": 0.0}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy artifact")

    def load(self, path):
        self._trained = True
'''

_DYNAMIC_SINR_SOURCE = _DYNAMIC_SOURCE.replace(
    'COMPONENT_NAME = "dl_bler"', 'COMPONENT_NAME = "cqi_quality"'
).replace('OUTPUT_FIELD = "dl_bler_pred"', 'OUTPUT_FIELD = "wb_cqi_avg_lag1_pred"').replace(
    "class RebuiltDlBler", "class CqiQuality"
)


def _register_dynamic_version(
    model_registry: ModelRegistry, component: str, model_class: str, output_field: str, source: str, tmp_path,
    adaptation_type: str = "regenerate", llm_metadata: dict | None = None,
):
    """Real `ModelRegistry.register_version_from_artifact()` call, writing a genuine source file
    and artifact to disk — exactly the shape a real RegenerationAgent/ExpandScopeAgent candidate
    leaves behind, so `_load_dynamic_class`'s real file I/O + dynamic import is genuinely
    exercised, not mocked."""
    source_path = tmp_path / f"{component}_source.py"
    source_path.write_text(source)
    artifact_path = tmp_path / f"{component}_artifact.bin"
    artifact_path.write_bytes(b"irrelevant bytes - this fake class's own load() ignores them")
    return model_registry.register_version_from_artifact(
        component=component, model_class=model_class, artifact_source_path=artifact_path,
        source_code_path=source_path, dependencies=(), feature_schema=("pusch_snr_avg_lag1",),
        output_field=output_field, adaptation_type=adaptation_type, parent_version_id=None,
        training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
        llm_metadata=llm_metadata,
    )


class _FakeD1Store:
    def __init__(self, history: pd.DataFrame) -> None:
        self._history = history

    def get_history(self, ue_id=None, cell_id=None):
        return self._history


def _bare_orchestrator() -> ContinuousOrchestrator:
    """Constructs the orchestrator WITHOUT calling `initialize()` — legitimate for testing the
    pure observation-construction/bootstrap-dispatch logic in isolation, since `__init__` sets up
    only plain state (no I/O, no telemetry/model construction). Also pre-populates the three
    dispatch dicts `initialize()` normally seeds before ever calling `_bootstrap_dt_models()`
    (`_target_columns`/`_output_fields`/`_dynamic_component_classes`), since a test calling
    `_bootstrap_dt_models()` directly skips that real setup step otherwise."""
    orch = ContinuousOrchestrator(SETTINGS, NO_KEY_SECRETS)
    orch._target_columns = dict(_TARGET_COLUMNS)
    orch._output_fields = dict(_OUTPUT_FIELDS)
    orch._dynamic_component_classes = {}
    return orch


# --- component spec consistency (the dispatch tables main.py itself relies on) --------------------


def test_component_spec_tables_are_internally_consistent():
    assert set(_DT_COMPONENT_CLASSES) == set(_TARGET_COLUMNS) == set(_OUTPUT_FIELDS)
    assert set(_DT_COMPONENT_CLASSES) == set(SETTINGS.drift.valid_components)
    for name, cls in _DT_COMPONENT_CLASSES.items():
        assert cls.COMPONENT_NAME == name
        assert _OUTPUT_FIELDS[name] == cls.OUTPUT_FIELD


# --- per-component previous-outcome tracking (feeds the Decision & Root-Cause Analysis Agent) ----


def test_previous_outcome_for_unknown_component_is_the_honest_default():
    orch = _bare_orchestrator()
    result = orch._previous_outcome_for("dl_bler")
    assert result == PreviousOutcome()  # no prior attempt on record — never fabricated


def test_previous_outcome_for_returns_the_tracked_value_for_that_component_only():
    orch = _bare_orchestrator()
    orch._previous_outcome_by_component["ul_bler"] = PreviousOutcome(
        action="recalibrate", verification_result="ACCEPT", fidelity_before=0.5, fidelity_after=0.9
    )

    assert orch._previous_outcome_for("ul_bler").action == "recalibrate"
    assert orch._previous_outcome_for("dl_prb") == PreviousOutcome()  # independent per component


# --- bootstrap dispatch: load existing vs. train new (no real training triggered here) ------------


class _FakeModelRegistry:
    """Stands in for `ModelRegistry` — only the methods `_bootstrap_dt_models` actually calls."""

    def __init__(self, existing_versions: dict[str, object]) -> None:
        self._existing = existing_versions
        self.loaded: list[str] = []

    def get_current_version(self, component: str):
        return self._existing.get(component)

    def load_artifact_into(self, instance, version) -> None:
        self.loaded.append(instance.COMPONENT_NAME)
        instance._trained = True  # noqa: SLF001 - test double simulating a real load

    def list_component_names(self) -> list[str]:
        return list(self._existing)


class _FakeVersion:
    def __init__(self, version_id: str, source_path: str | None = None) -> None:
        self.version_id = version_id
        self.source_path = source_path  # None = statically-imported class (bootstrap/recalibrate)


def test_bootstrap_dt_models_loads_existing_versions_without_generating_bootstrap_data():
    orch = _bare_orchestrator()
    fake_versions = {name: _FakeVersion(f"{name}-v1") for name in _DT_COMPONENT_CLASSES}  # every component already has a production version
    orch.model_registry = _FakeModelRegistry(fake_versions)

    def _fail_if_called():
        raise AssertionError("bootstrap telemetry must not be generated when every component already has a production version")

    orch._generate_bootstrap_history = _fail_if_called  # type: ignore[method-assign]
    orch._bootstrap_dt_models()

    assert set(orch.model_registry.loaded) == set(_DT_COMPONENT_CLASSES)
    assert set(orch._dt_components) == set(_DT_COMPONENT_CLASSES)
    for instance in orch._dt_components.values():
        assert instance.is_trained


# --- hot swap: making an ACCEPT take over live serving without a restart --------------------------


def test_component_class_for_unknown_name_returns_none():
    orch = _bare_orchestrator()
    assert orch._component_class_for("not_a_real_component") is None


def test_component_class_for_known_static_component():
    orch = _bare_orchestrator()
    assert orch._component_class_for("dl_bler") is _DT_COMPONENT_CLASSES["dl_bler"]


def test_component_class_for_prefers_a_dynamically_loaded_class_once_one_exists():
    orch = _bare_orchestrator()

    class _FakeDynamicThroughput:
        pass

    orch._dynamic_component_classes["dl_bler"] = _FakeDynamicThroughput
    assert orch._component_class_for("dl_bler") is _FakeDynamicThroughput


def test_hot_swap_recalibrate_replaces_live_instance_with_the_trained_candidate():
    orch = _bare_orchestrator()
    orch.dt_model_registry = DTModelRegistry()
    original = DlBlerModel.from_settings(SETTINGS)
    orch.dt_model_registry.register(original)

    candidate = DlBlerModel.from_settings(SETTINGS)  # a second, distinct trained instance
    agent_result = SimpleNamespace(
        version=SimpleNamespace(version_id="dl_bler-v2", model_class="DlBlerModel"),
        candidate_component=candidate,
    )
    orch._hot_swap_candidate("recalibrate", "dl_bler", agent_result)

    assert orch.dt_model_registry.get("dl_bler") is candidate
    assert orch.dt_model_registry.get("dl_bler") is not original
    # recalibrate never introduces a new class — the dynamic-class table stays untouched.
    assert "dl_bler" not in orch._dynamic_component_classes


def test_hot_swap_regenerate_dynamically_loads_the_new_class_and_serves_it(tmp_path):
    orch = _bare_orchestrator()
    orch.dt_model_registry = DTModelRegistry()
    orch.dt_model_registry.register(DlBlerModel.from_settings(SETTINGS))
    orch.model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")

    version = _register_dynamic_version(
        orch.model_registry, "dl_bler", "RebuiltDlBler", "dl_bler_pred", _DYNAMIC_SOURCE, tmp_path,
    )
    agent_result = SimpleNamespace(version=version)
    orch._hot_swap_candidate("regenerate", "dl_bler", agent_result)

    live = orch.dt_model_registry.get("dl_bler")
    assert type(live).__name__ == "RebuiltDlBler"
    assert live.is_trained  # load() was genuinely called
    assert orch._dynamic_component_classes["dl_bler"].__name__ == "RebuiltDlBler"
    # And the swapped-in class is genuinely what SERVES predictions now, not just registered —
    # go through a real DTOrchestrator over the live registry, exactly like run_prediction_and_
    # fidelity_cycle() does, rather than calling the new instance directly.
    from src.dt_models.orchestrator import DTOrchestrator

    features = pd.DataFrame({"pusch_snr_avg_lag1": [10.0, 20.0]})
    preds = DTOrchestrator(orch.dt_model_registry).run_predictions(features)["dl_bler"]
    assert list(preds) == [999.0, 999.0]


def test_hot_swap_expand_scope_registers_a_genuinely_new_component(tmp_path):
    orch = _bare_orchestrator()
    orch.dt_model_registry = DTModelRegistry()
    orch.dt_model_registry.register(DlBlerModel.from_settings(SETTINGS))  # the one pre-existing component
    orch.model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")

    version = _register_dynamic_version(
        orch.model_registry, "cqi_quality", "CqiQuality", "wb_cqi_avg_lag1_pred", _DYNAMIC_SINR_SOURCE, tmp_path,
        adaptation_type="expand_scope",
    )
    agent_result = SimpleNamespace(version=version, design=SimpleNamespace(target_column="wb_cqi_avg_lag1"))
    assert "cqi_quality" not in orch.dt_model_registry  # genuinely new — never registered before

    orch._hot_swap_candidate("expand_scope", "cqi_quality", agent_result)

    assert "cqi_quality" in orch.dt_model_registry
    assert type(orch.dt_model_registry.get("cqi_quality")).__name__ == "CqiQuality"
    assert orch._target_columns["cqi_quality"] == "wb_cqi_avg_lag1"
    assert orch._output_fields["cqi_quality"] == "wb_cqi_avg_lag1_pred"
    assert "dl_bler" in orch.dt_model_registry  # the pre-existing component is untouched


def test_bootstrap_dt_models_restores_a_regenerated_component_using_its_actual_class(tmp_path):
    """A component whose CURRENT production version came from a prior regenerate (source_path is
    set) must be loaded as that dynamically-generated class at boot — never the original static
    one, which would silently mix an unrelated class's weights into the wrong implementation."""
    orch = _bare_orchestrator()
    orch.model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    for name in _DT_COMPONENT_CLASSES:
        if name == "dl_bler":
            continue
        orch.model_registry.register_version(
            component_instance=_DT_COMPONENT_CLASSES[name].from_settings(SETTINGS), adaptation_type="bootstrap",
            parent_version_id=None, training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
        )
    _register_dynamic_version(orch.model_registry, "dl_bler", "RebuiltDlBler", "dl_bler_pred", _DYNAMIC_SOURCE, tmp_path)

    orch._bootstrap_dt_models()

    assert type(orch._dt_components["dl_bler"]).__name__ == "RebuiltDlBler"
    assert orch._dynamic_component_classes["dl_bler"].__name__ == "RebuiltDlBler"
    # the other four are untouched — still the original static classes.
    for name in _DT_COMPONENT_CLASSES:
        if name != "dl_bler":
            assert type(orch._dt_components[name]) is _DT_COMPONENT_CLASSES[name]


def test_bootstrap_dt_models_restores_an_extra_expand_scope_component_across_a_restart(tmp_path):
    orch = _bare_orchestrator()
    orch.model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    for name in _DT_COMPONENT_CLASSES:
        orch.model_registry.register_version(
            component_instance=_DT_COMPONENT_CLASSES[name].from_settings(SETTINGS), adaptation_type="bootstrap",
            parent_version_id=None, training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
        )
    _register_dynamic_version(
        orch.model_registry, "cqi_quality", "CqiQuality", "wb_cqi_avg_lag1_pred", _DYNAMIC_SINR_SOURCE, tmp_path,
        adaptation_type="expand_scope", llm_metadata={"target_column": "wb_cqi_avg_lag1"},
    )

    orch._bootstrap_dt_models()

    assert "cqi_quality" in orch._dt_components
    assert type(orch._dt_components["cqi_quality"]).__name__ == "CqiQuality"
    assert orch._target_columns["cqi_quality"] == "wb_cqi_avg_lag1"


def test_bootstrap_dt_models_warns_but_does_not_crash_when_target_column_metadata_is_missing(tmp_path):
    orch = _bare_orchestrator()
    orch.model_registry = ModelRegistry(models_dir=tmp_path / "models", index_path=tmp_path / "models" / "index.json")
    for name in _DT_COMPONENT_CLASSES:
        orch.model_registry.register_version(
            component_instance=_DT_COMPONENT_CLASSES[name].from_settings(SETTINGS), adaptation_type="bootstrap",
            parent_version_id=None, training_window={}, evaluation_window={}, evaluation_metrics={}, status="production",
        )
    _register_dynamic_version(  # no llm_metadata at all
        orch.model_registry, "cqi_quality", "CqiQuality", "wb_cqi_avg_lag1_pred", _DYNAMIC_SINR_SOURCE, tmp_path,
        adaptation_type="expand_scope",
    )

    orch._bootstrap_dt_models()  # must not raise

    assert "cqi_quality" in orch._dt_components
    assert "cqi_quality" not in orch._target_columns
