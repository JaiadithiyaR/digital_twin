"""DL PRB component — OAI migration (prompt2.md Step 5). Replaces the retired prb_utilization
component. Predicts downlink PRBs used per second (RRU.PrbTotDl, raw PRB count).

Root node in the dependency graph (no upstream DT predictions required) — inputs are the full
lagged OAI feature vector (`src/data/oai_dataset.ALL_FEATURES`), read directly from D1.
`dl_throughput` depends on this component's prediction.

Phase 13 §1.2: learns the residual `Δy = y_t - y_{t-1}` (see `src/dt_models/residual_base.py`);
`predict()` still returns `ŷ_t = y_{t-1} + Δŷ` — the public interface is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.oai_dataset import ALL_FEATURES
from src.dt_models.residual_base import ResidualDTComponent

if TYPE_CHECKING:
    from src.common.config import Settings


class DlPrbModel(ResidualDTComponent):
    COMPONENT_NAME = "dl_prb"
    DEPENDENCIES: tuple[str, ...] = ()
    REQUIRED_FEATURES = ALL_FEATURES
    OUTPUT_FIELD = "dl_prb_pred"

    @classmethod
    def from_settings(cls, settings: "Settings") -> "DlPrbModel":
        spec = settings.dt_models.dl_prb
        return cls(model_type=spec.model_type, params=spec.params)
