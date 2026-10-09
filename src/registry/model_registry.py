"""Versioned model/component registry — tracks model VERSIONS per component (every artifact ever
produced, which one is production, full rollback history). Three registries exist in this
codebase, do not confuse them: `dt_models/component_registry.py` catalogs D1's state tables;
`dt_models/model_registry.py`'s `DTModelRegistry` catalogs live in-memory prediction components
for the orchestrator (no versioning); this file tracks versions, independent of D1 — promoting a
version here never touches D1 and vice versa.

Every adaptation agent registers a new artifact here via `register_version()`.
`promote()`/`reject()` are called only by the Verification Agent — nothing here auto-invokes
them, so a registered version is never implicitly trusted.

Persistence: one JSON index (`<models_dir>/registry_index.json`, atomic write) + one artifact
file per version (`<models_dir>/<component>/<version_id>.joblib`).
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
    """Identifies one candidate/production artifact. Immutable (`frozen`) — `status` changes
    only by replacing the record wholesale (`model_copy`) via `promote()`/`reject()`, never by
    in-place mutation, so a superseded record stays bit-for-bit what was actually promoted."""

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
        """Persist `component_instance`'s trained state as a new immutable version. Never mutates
        an existing version; `status` is always whatever the caller passes — this registry never
        promotes anything on its own."""
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
        """`register_version()`'s counterpart for sandboxed, LLM-generated candidates: the trained
        artifact already exists as a file the sandbox wrote, so this only COPIES it (+ source, for
        audit) in — it never imports the untrusted class. Because the class was never imported
        here, this registry can't reconstruct a live instance either; `source_path()` is the
        read-only hook a caller with its own vetted loading path (the hot-swap mechanism, only
        after a real ACCEPT) uses to do so."""
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
        """Marks a candidate `"rejected"` — retained for audit, never deleted, never affects
        production."""
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
        """Reverts to the most recently superseded version. Raises if there's nothing to roll
        back to."""
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
        """Restore `version`'s trained artifact into an already-constructed instance of the
        matching class — this registry never constructs instances itself."""
        component_instance.load(self._models_dir / version.artifact_path)

    def artifact_path(self, version: ModelVersionMetadata) -> Path:
        """Resolve `version`'s artifact to an absolute path — read-only, no deserialization.
        Used by Verification to confirm the artifact exists on disk without unpickling it."""
        return self._models_dir / version.artifact_path

    def source_path(self, version: ModelVersionMetadata) -> Path | None:
        """Resolve `version`'s saved LLM-generated source file to an absolute path (`None` for a
        recalibration/bootstrap version — no source file of its own). Read-only; never imports."""
        if version.source_path is None:
            return None
        return self._models_dir / version.source_path

    def load_dynamic_class(self, version: ModelVersionMetadata) -> type:
        """Imports a regenerate/expand_scope candidate's LLM-generated class from its saved
        source — the one point this trusted process imports LLM code, only ever for a version
        already passed by the sandbox and verification."""
        import importlib.util

        source_path = self.source_path(version)
        if source_path is None:
            raise RuntimeError(f"version {version.version_id!r} of {version.component!r} has no saved source to load")
        spec = importlib.util.spec_from_file_location(f"_dt_dynamic_{version.component}_{version.version_id}", source_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, version.model_class)

    def list_component_names(self) -> list[str]:
        """Every component name this registry has ever tracked a version for, including one
        produced by expand_scope. Used at boot to restore such components."""
        return list(self._versions.keys())
