#!/usr/bin/env python3
"""phase13 §5 item 9 — `model_vs_baselines_<component>.png`: RMSE of the model vs. persistence
vs. the 5-second moving average, per CV fold, one figure per component. Reads ONLY the already-
written `reports/model_eval/model_eval.csv` (produced by `scripts/bootstrap_oai_models.py`) —
never re-runs training. One figure per file, PNG 300dpi + PDF, titled/labeled axes with units,
legended, plus a CSV of exactly the plotted data alongside every graph (phase13 §5's own rules).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_EVAL_CSV = REPO_ROOT / "reports" / "model_eval" / "model_eval.csv"
GRAPHS_DIR = REPO_ROOT / "reports" / "model_eval" / "graphs"


def main() -> None:
    df = pd.read_csv(MODEL_EVAL_CSV)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for component, sub in df.groupby("component"):
        sub = sub.sort_values("held_out_run")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(sub))
        width = 0.25
        ax.bar([i - width for i in x], sub["model_rmse"], width, label="model (residual)", color="tab:blue")
        ax.bar(list(x), sub["persistence_rmse"], width, label="persistence (y_t-1)", color="tab:gray")
        ax.bar([i + width for i in x], sub["ma5s_rmse"], width, label="5s moving average", color="tab:orange")
        ax.set_xticks(list(x), [f"held-out run {r}" for r in sub["held_out_run"]])
        ax.set_ylabel(f"RMSE ({component}, raw unit — see docs/MODELS.md for display unit)")
        ax.set_title(f"Model vs. baselines — {component} (leave-one-run-out CV)")
        ax.legend()
        fig.tight_layout()
        out_path = GRAPHS_DIR / f"model_vs_baselines_{component}.png"
        fig.savefig(out_path, dpi=300)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        sub[["held_out_run", "model_rmse", "persistence_rmse", "ma5s_rmse"]].to_csv(out_path.with_suffix(".csv"), index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
