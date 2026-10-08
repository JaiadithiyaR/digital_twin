"""DL RLC delay component — OAI migration (prompt2.md Step 5). Replaces the retired `latency`
component. Predicts `DRB.RlcSduDelayDl` (ms) for second t — sourced, per the real export (see
`src/data/oai_dataset.py`'s own docstring), from `rlc_bearer_1s.tx_sdu_avg_time_to_tx_avg`.

Depends on `dl_throughput` and `dl_bler` (prompt2.md's dependency graph: "dl_rlc_delay ←
dl_throughput, dl_bler") — the orchestrator supplies both dependencies' live predictions at
inference time; direct/bootstrap training supplies their ground-truth historical values under
the same column names (the standard "train on ground truth, serve on upstream predictions"
pattern already established by the retired `LatencyModel`/`JitterModel`). Phase 13 §1.4:
downstream components receive the PREDICTED upstream value for t (`{dep}_pred`, supplied below)
AND the MEASURED upstream value for t-1 (`{dep}_lag1`, already part of `ALL_FEATURES`/
`REQUIRED_FEATURES`) — both simultaneously, not one or the other.

Phase 13 §1.2: learns the residual `Δy = y_t - y_{t-1}` (see `src/dt_models/residual_base.py`);
`predict()` still returns `ŷ_t = y_{t-1} + Δŷ` — the public interface is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.oai_dataset import ALL_FEATURES
from src.dt_models.residual_base import ResidualDTComponent

if TYPE_CHECKING:
    from src.common.config import Settings


class DlRlcDelayModel(ResidualDTComponent):
    COMPONENT_NAME = "dl_rlc_delay"
    DEPENDENCIES = ("dl_throughput", "dl_bler")
    REQUIRED_FEATURES = ALL_FEATURES
    OUTPUT_FIELD = "dl_rlc_delay_pred"

    DL_BLER_INPUT_COLUMN = "dl_bler_pred"
    DL_THROUGHPUT_INPUT_COLUMN = "dl_throughput_pred"

    @property
    def _input_columns(self) -> tuple[str, ...]:
        return tuple(self.REQUIRED_FEATURES) + (self.DL_BLER_INPUT_COLUMN, self.DL_THROUGHPUT_INPUT_COLUMN)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "DlRlcDelayModel":
        spec = settings.dt_models.dl_rlc_delay
        return cls(model_type=spec.model_type, params=spec.params)
