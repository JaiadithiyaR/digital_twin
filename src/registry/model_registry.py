"""Versioned model/component registry (prompt.md §46, §29) — Concept C from CLAUDE.md §4.

This is the third, distinct "registry" in this codebase — do not confuse it with the other two:

1. `src/dt_models/component_registry.py`'s `ComponentRegistry` (Module 4/D1) — catalogs D1's own
   state TABLES (current/history/quarantine). Nothing to do with prediction models.
2. `src/dt_models/model_registry.py`'s `DTModelRegistry` (Module 5) — catalogs which prediction
   COMPONENTS exist and their dependency graph, purely in-memory, no versioning, so the
   orchestrator can compute an execution order. A component registered there is a single live
   instance the orchestrator runs predictions through.
3. **`ModelRegistry` (this file)** — tracks MODEL **VERSIONS** per component: every trained
   artifact ever produced (bootstrap, recalibrated, regenerated, expand-scope), which one is
   currently in production, and the full history needed for rollback. This is Concept C
   (CLAUDE.md §4): "independently versioned learned prediction implementations," deliberately
   separate from Concept B (D1's continuously-updated dynamic state) — promoting a model version
   here never touches D1, and D1 being updated never touches this registry (prompt.md §0.5).

Needed by Module 14 (Recalibration, this turn) and, unchanged, by Modules 15 (Regeneration) and
16 (Expand-Scope) later — every adaptation agent that produces a new artifact registers it here
the same way, via `register_version()`. Promotion/rejection (deciding whether a candidate
version's status becomes `"production"` or `"rejected"`) is Module 17 (Agentic Verification)'s
job, not built yet — `promote()`/`reject()` exist here as registry primitives Module 17 will call,
but nothing in this codebase invokes them automatically; a version this registry creates is never
implicitly trusted (prompt.md §46: "must never allow an invalid candidate to become active").

Persistence: one JSON index file (`<models_dir>/registry_index.json`, atomic write-temp-then-
rename, mirroring `D1Store`'s `_atomic_write_parquet` pattern) holding every `ModelVersionMetadata`
record ever created, plus one artifact file per version
(`<models_dir>/<component>/<version_id>.joblib`, written via the component's own already-tested
`DTComponent.save()`). JSON (not Parquet) is the right fit here — a registry index is a modest
number of richly-structured, human-inspectable records, not a large row-oriented telemetry table.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from src.common.config import Settings
    from src.dt_models.base import DTComponent

logger = logging.getLogger(__name__)


class ModelRegistryError(Exception):
    """Raised for any invalid registry operation (unknown component/version, promoting something
    that doesn't exist, etc.) — never silently ignored or guessed around."""


class ModelVersionMetadata(BaseModel):
    """Everything prompt.md §29 requires to identify one candidate/production artifact.

    Immutable once created (`model_config = frozen`) — the ONLY field that ever changes after
    creation is `status`, and that happens by replacing the record wholesale (`model_copy`) via
    `promote()`/`reject()`/`supersede`, never by mutating a stored instance in place. This keeps
    every version's history genuinely auditable: a `"production"` record retained after being
    superseded is bit-for-bit what was actually promoted at the time.
    """

    model_config = ConfigDict(frozen=True)

    component: str
    version_id: str
    model_class: str  # the concrete DTComponent subclass name (e.g. "ThroughputModel") — for
    # human/audit legibility; reconstructing an instance is the caller's job (they already know
    # which class, since they supplied the trained instance to `register_version`), not this
    # registry's — see module docstring's separation from `DTModelRegistry`.
    artifact_path: str  # relative to `models_dir`, e.g. "throughput/throughput-v2.joblib"
    source_path: str | None = None  # relative to `models_dir` — LLM-generated candidate source
    # (Modules 15/16), retained for audit (prompt.md §29 "source code/artifact location"); None
    # for recalibration/bootstrap versions, which reuse the already-trusted production class
    # rather than new source.
    dependencies: tuple[str, ...]
    feature_schema: tuple[str, ...]
    output_field: str
    adaptation_type: Literal["bootstrap", "recalibrate", "regenerate", "expand_scope"]
    parent_version_id: str | None
    created_at: datetime
    training_window: dict[str, Any]
    evaluation_window: dict[str, Any]
    evaluation_metrics: dict[str, float]
    fidelity_before: float | None = None
    fidelity_after: float | None = None
    status: Literal["candidate", "production", "rejected", "superseded"]
    llm_metadata: dict[str, Any] | None = None


def _atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp_path, path)


class ModelRegistry:
    """Thread-safe (matching `D1Store`'s discipline — an adaptation agent's registration must
    never race a concurrent one for a different component, or a future reader)."""

    def __init__(self, models_dir: Path, index_path: Path) -> None:
        self._models_dir = models_dir
        self._index_path = index_path
        self._lock = threading.Lock()
        self._versions: dict[str, list[ModelVersionMetadata]] = self._load_index()

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ModelRegistry":
        models_dir = settings.resolve_path(settings.storage.models_dir)
        return cls(models_dir=models_dir, index_path=models_dir / "registry_index.json")

    def _load_index(self) -> dict[str, list[ModelVersionMetadata]]:
        if not self._index_path.exists():
            return {}
        raw = json.loads(self._index_path.read_text())
        return {
            component: [ModelVersionMetadata.model_validate(record) for record in records]
            for component, records in raw.items()
        }

    def _write_index_locked(self) -> None:
        payload = {
            component: [json.loads(v.model_dump_json()) for v in versions]
            for component, versions in self._versions.items()
        }
        _atomic_write_json(payload, self._index_path)

    # --- write path ------------------------------------------------------------------------------

    def register_version(
        self,
        *,
        component_instance: "DTComponent",
        adaptation_type: Literal["bootstrap", "recalibrate", "regenerate", "expand_scope"],
        parent_version_id: str | None,
        training_window: dict[str, Any],
        evaluation_window: dict[str, Any],
        evaluation_metrics: dict[str, float],
        fidelity_before: float | None = None,
        fidelity_after: float | None = None,
        status: Literal["candidate", "production", "rejected", "superseded"] = "candidate",
        llm_metadata: dict[str, Any] | None = None,
    ) -> ModelVersionMetadata:
        """Persist `component_instance`'s trained state as a new immutable version and record its
        metadata. Never mutates any existing version, and never itself decides `status` beyond
        what the caller explicitly passes — a candidate is `"candidate"` unless the caller (e.g.
        a bootstrap-training step registering the very first version) explicitly says otherwise;
        this registry never promotes anything on its own (prompt.md §46)."""
        component = component_instance.COMPONENT_NAME
        with self._lock:
            existing = self._versions.get(component, [])
            version_id = f"{component}-v{len(existing) + 1}"
            artifact_rel_path = f"{component}/{version_id}.joblib"
            component_instance.save(self._models_dir / artifact_rel_path)

            metadata = ModelVersionMetadata(
                component=component,
                version_id=version_id,
                model_class=type(component_instance).__name__,
                artifact_path=artifact_rel_path,
                dependencies=tuple(component_instance.DEPENDENCIES),
                feature_schema=tuple(component_instance.REQUIRED_FEATURES),
                output_field=component_instance.OUTPUT_FIELD,
                adaptation_type=adaptation_type,
                parent_version_id=parent_version_id,
                created_at=datetime.now(UTC),
                training_window=training_window,
                evaluation_window=evaluation_window,
                evaluation_metrics=evaluation_metrics,
                fidelity_before=fidelity_before,
                fidelity_after=fidelity_after,
                status=status,
                llm_metadata=llm_metadata,
            )
            self._versions.setdefault(component, []).append(metadata)
            self._write_index_locked()

        logger.info(
            "model version registered",
            extra={
                "component": "model_registry",
                "component_name": component,
                "version_id": version_id,
                "adaptation_type": adaptation_type,
                "status": status,
                "parent_version_id": parent_version_id,
            },
        )
        return metadata

    def register_version_from_artifact(
        self,
        *,
        component: str,
        model_class: str,
        artifact_source_path: Path,
        source_code_path: Path | None,
        dependencies: tuple[str, ...],
        feature_schema: tuple[str, ...],
        output_field: str,
        adaptation_type: Literal["bootstrap", "recalibrate", "regenerate", "expand_scope"],
        parent_version_id: str | None,
        training_window: dict[str, Any],
        evaluation_window: dict[str, Any],
        evaluation_metrics: dict[str, float],
        fidelity_before: float | None = None,
        fidelity_after: float | None = None,
        status: Literal["candidate", "production", "rejected", "superseded"] = "candidate",
        llm_metadata: dict[str, Any] | None = None,
    ) -> ModelVersionMetadata:
        """`register_version()`'s counterpart for candidates produced OUTSIDE this process
        (Modules 15/16's LLM-generated, sandboxed code — `src/sandbox/executor.py`). The trained
        artifact already exists as a file the sandbox subprocess wrote; this process never
        imports the untrusted class to call `.save()` on it the way `register_version()` does for
        an already-trusted `DTComponent` instance — that would defeat the sandbox boundary
        entirely. Instead this just COPIES the already-produced artifact (and, if given, the
        candidate's source for audit — prompt.md §29) into this registry's managed storage.

        Because the candidate's class was never imported here, this registry cannot later
        reconstruct a live instance of it either (`load_artifact_into` needs an
        already-constructed instance of the matching class, which no code in this process can
        safely produce for LLM-generated candidates) — promoting a regenerated/expand-scope
        version to actually SERVE production predictions is therefore a follow-up concern for
        whenever Module 17 (verification) and a vetted dynamic-loading path exist, out of scope
        here. This method only ever creates `"candidate"`-or-explicitly-stated-status records.
        """
        with self._lock:
            existing = self._versions.get(component, [])
            version_id = f"{component}-v{len(existing) + 1}"

            artifact_rel_path = f"{component}/{version_id}.joblib"
            artifact_dest = self._models_dir / artifact_rel_path
            artifact_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_source_path, artifact_dest)

            source_rel_path: str | None = None
            if source_code_path is not None:
                source_rel_path = f"{component}/{version_id}.py"
                shutil.copyfile(source_code_path, self._models_dir / source_rel_path)

            metadata = ModelVersionMetadata(
                component=component,
                version_id=version_id,
                model_class=model_class,
                artifact_path=artifact_rel_path,
                source_path=source_rel_path,
                dependencies=tuple(dependencies),
                feature_schema=tuple(feature_schema),
                output_field=output_field,
                adaptation_type=adaptation_type,
                parent_version_id=parent_version_id,
                created_at=datetime.now(UTC),
                training_window=training_window,
                evaluation_window=evaluation_window,
                evaluation_metrics=evaluation_metrics,
                fidelity_before=fidelity_before,
                fidelity_after=fidelity_after,
                status=status,
                llm_metadata=llm_metadata,
            )
            self._versions.setdefault(component, []).append(metadata)
            self._write_index_locked()

        logger.info(
            "model version registered from sandboxed artifact",
            extra={
                "component": "model_registry",
                "component_name": component,
                "version_id": version_id,
                "adaptation_type": adaptation_type,
                "status": status,
                "parent_version_id": parent_version_id,
            },
        )
        return metadata

    def promote(self, component: str, version_id: str) -> ModelVersionMetadata:
        """Atomically makes `version_id` the production version for `component`. Any prior
        `"production"` record is demoted to `"superseded"` and RETAINED (never deleted) — rollback
        safety per prompt.md §0.21/CLAUDE.md §8: "previous known-good versions are always
        retained for rollback." Only one version can be `"production"` per component at a time,
        by construction (this loop touches every record for the component atomically)."""
        with self._lock:
            versions = self._versions.get(component, [])
            if not any(v.version_id == version_id for v in versions):
                raise ModelRegistryError(f"no version {version_id!r} registered for component {component!r}")
            updated: list[ModelVersionMetadata] = []
            target: ModelVersionMetadata | None = None
            for v in versions:
                if v.version_id == version_id:
                    target = v.model_copy(update={"status": "production"})
                    updated.append(target)
                elif v.status == "production":
                    updated.append(v.model_copy(update={"status": "superseded"}))
                else:
                    updated.append(v)
            self._versions[component] = updated
            self._write_index_locked()
        assert target is not None
        logger.info(
            "model version promoted",
            extra={"component": "model_registry", "component_name": component, "version_id": version_id},
        )
        return target

    def reject(self, component: str, version_id: str) -> ModelVersionMetadata:
        """Marks a candidate as `"rejected"` — retained for audit (prompt.md §8's "never silently
        discard" principle, extended here), never deleted, never affects production."""
        with self._lock:
            versions = self._versions.get(component, [])
            updated = []
            target: ModelVersionMetadata | None = None
            for v in versions:
                if v.version_id == version_id:
                    target = v.model_copy(update={"status": "rejected"})
                    updated.append(target)
                else:
                    updated.append(v)
            if target is None:
                raise ModelRegistryError(f"no version {version_id!r} registered for component {component!r}")
            self._versions[component] = updated
            self._write_index_locked()
        logger.info(
            "model version rejected",
            extra={"component": "model_registry", "component_name": component, "version_id": version_id},
        )
        return target

    def rollback(self, component: str) -> ModelVersionMetadata:
        """Reverts to the most recently superseded version, promoting it back to `"production"`
        and demoting the current production version to `"superseded"` — the rollback safety
        prompt.md §0.21 requires. Raises if there is nothing to roll back to."""
        with self._lock:
            versions = self._versions.get(component, [])
            superseded = [v for v in versions if v.status == "superseded"]
            if not superseded:
                raise ModelRegistryError(f"no superseded version to roll back to for component {component!r}")
            rollback_target = superseded[-1]  # most recently superseded (list is append-ordered)
        return self.promote(component, rollback_target.version_id)

    # --- read path -------------------------------------------------------------------------------

    def get_current_version(self, component: str) -> ModelVersionMetadata | None:
        for v in reversed(self._versions.get(component, [])):
            if v.status == "production":
                return v
        return None

    def get_version(self, component: str, version_id: str) -> ModelVersionMetadata:
        for v in self._versions.get(component, []):
            if v.version_id == version_id:
                return v
        raise ModelRegistryError(f"no version {version_id!r} registered for component {component!r}")

    def list_versions(self, component: str) -> list[ModelVersionMetadata]:
        return list(self._versions.get(component, []))

    def load_artifact_into(self, component_instance: "DTComponent", version: ModelVersionMetadata) -> None:
        """Restore `version`'s trained artifact into `component_instance` (already constructed,
        of the matching class — see module docstring: this registry doesn't reconstruct
        instances itself, only tracks/persists what a caller already built and trained)."""
        component_instance.load(self._models_dir / version.artifact_path)

    def artifact_path(self, version: ModelVersionMetadata) -> Path:
        """Resolve `version`'s artifact to an absolute filesystem path — read-only path
        resolution, no deserialization. Used by Module 17 (Agentic Verification) to confirm a
        candidate's artifact genuinely exists on disk (prompt.md §32 "candidate successfully
        loads") without importing or unpickling it — this registry never trusts an artifact's
        bytes as executable in this process (see `register_version_from_artifact`'s docstring)."""
        return self._models_dir / version.artifact_path
