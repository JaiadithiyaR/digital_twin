"""DL BLER component — OAI migration (prompt2.md Step 5). Replaces the retired packet_loss
component. Predicts the downlink MAC block-error rate (0-1 ratio) for second t.

Root node in the dependency graph (no upstream DT predictions required) — inputs are the full
lagged OAI feature vector (`src/data/oai_dataset.ALL_FEATURES`), read directly from D1.
`dl_throughput` and `dl_rlc_delay` both depend on this component's prediction.

Phase 13 §1.2: learns the residual `Δy = y_t - y_{t-1}` (see `src/dt_models/residual_base.py`);
`predict()` still returns `ŷ_t = y_{t-1} + Δŷ` — the public interface is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.oai_dataset import ALL_FEATURES
from src.dt_models.residual_base import ResidualDTComponent

if TYPE_CHECKING:
    from src.common.config import Settings


class DlBlerModel(ResidualDTComponent):
    COMPONENT_NAME = "dl_bler"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = ALL_FEATURES
    # Deliberately distinct from D1's ground-truth "dl_bler" column so a downstream component
    # can never confuse a prediction for the raw telemetry it was trained against.
    OUTPUT_FIELD = "dl_bler_pred"

    @classmethod
    def from_settings(cls, settings: "Settings") -> "DlBlerModel":
        spec = settings.dt_models.dl_bler
        return cls(model_type=spec.model_type, params=spec.params)
