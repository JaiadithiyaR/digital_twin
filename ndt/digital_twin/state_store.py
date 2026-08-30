import json
from pathlib import Path
import pandas as pd

# ============================================================
# MODULE 4
# Digital Twin State and History Store
# ============================================================

BASE_DIR = Path.home() / "NDT"

TELEMETRY_FILE = (
    BASE_DIR
    / "ndt"
    / "data"
    / "telemetry_cleaned.csv"
)

CURRENT_STATE_FILE = (
    BASE_DIR
    / "ndt"
    / "data"
    / "dt_current_state.json"
)

HISTORY_FILE = (
    BASE_DIR
    / "ndt"
    / "data"
    / "dt_history.csv"
)


# ============================================================
# Load current synchronized state
# ============================================================

def load_current_state():
    
    if not CURRENT_STATE_FILE.exists():

        raise FileNotFoundError(
            "Current DT state not found:\n"
            f"{CURRENT_STATE_FILE}\n\n"
            "Run Module 3 first."
        )

    with open(
        CURRENT_STATE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# Update DT history
# ============================================================

def update_history():

    state = load_current_state()

    timestamp = state[
        "synchronization_timestamp"
    ]

    ue_states = state["ue_states"]

    records = []

    for ue_id, values in ue_states.items():

        record = values.copy()

        record["dt_sync_timestamp"] = timestamp

        records.append(record)

    if not records:
        print("[DT] No UE states available.")
        return

    new_data = pd.DataFrame(records)

    # --------------------------------------------------------
    # Append to existing DT history
    # --------------------------------------------------------

    if HISTORY_FILE.exists():

        old_data = pd.read_csv(
            HISTORY_FILE
        )

        combined = pd.concat(
            [
                old_data,
                new_data
            ],
            ignore_index=True
        )

    else:

        combined = new_data

    # --------------------------------------------------------
    # Remove duplicate DT states
    #
    # One UE should have one record for a given
    # synchronization timestamp.
    # --------------------------------------------------------

    combined = combined.drop_duplicates(
        subset=[
            "dt_sync_timestamp",
            "ue_id"
        ],
        keep="last"
    )

    # --------------------------------------------------------
    # Maintain chronological time-series order
    # --------------------------------------------------------

    combined = combined.sort_values(
        by=[
            "dt_sync_timestamp",
            "ue_id"
        ],
        kind="stable"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        HISTORY_FILE,
        index=False,
        float_format="%.6f"
    )

    print(
        f"[DT] History updated: "
        f"{len(combined)} rows"
    )


# ============================================================
# Display current DT state
# ============================================================

def show_current_state():

    state = load_current_state()

    print()
    print("==============================================")
    print(" CURRENT DIGITAL TWIN STATE")
    print("==============================================")

    print(
        f"Timestamp : "
        f"{state['synchronization_timestamp']}"
    )

    print(
        f"UE count  : "
        f"{state['network']['ue_count']}"
    )

    print(
        f"gNB count : "
        f"{state['network']['gnb_count']}"
    )

    for ue_id, values in state[
        "ue_states"
    ].items():

        print()
        print(f"UE {ue_id}")

        print(
            f"  Throughput : "
            f"{values['throughput_mbps']} Mbps"
        )

        print(
            f"  Latency    : "
            f"{values['latency_ms']} ms"
        )

        print(
            f"  Jitter     : "
            f"{values['jitter_ms']} ms"
        )

        print(
            f"  SINR       : "
            f"{values['sinr_db']} dB"
        )

        print(
            f"  RSRP       : "
            f"{values['rsrp_dbm']} dBm"
        )

        print(
            f"  PRB usage  : "
            f"{values['prb_utilization_percent']} %"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("==============================================")
    print(" NDT MODULE 4")
    print(" DT State + History Store")
    print("==============================================")

    update_history()

    show_current_state()

    print()
    print(
        f"[DT] History file: {HISTORY_FILE}"
    )


if __name__ == "__main__":
    main()