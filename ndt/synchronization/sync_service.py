#!/usr/bin/env python3

import json
import os
import time
from pathlib import Path

import pandas as pd


# ============================================================
# MODULE 3
# Continuous Digital Twin Synchronization
# ============================================================

BASE_DIR = Path.home() / "NDT"

INPUT_FILE = (
    BASE_DIR
    / "ndt"
    / "data"
    / "telemetry_cleaned.csv"
)

STATE_FILE = (
    BASE_DIR
    / "ndt"
    / "data"
    / "dt_current_state.json"
)

SYNC_INTERVAL = 1.0


# ============================================================
# Required telemetry columns
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "ue_id",
    "cell_id",
    "ue_count",
    "gnb_count",
    "ue_x",
    "ue_y",
    "ue_z",
    "ue_speed_mps",
    "throughput_mbps",
    "offered_load_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_percent",
    "packet_delivery_ratio_percent",
    "tx_packets",
    "rx_packets",
    "tx_bytes",
    "rx_bytes",
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "cqi",
    "mcs",
    "ri",
    "prb_utilization_percent",
    "symbol_utilization_percent",
    "scheduled_ues",
    "ue_tx_power_dbm",
]


# ============================================================
# Validate input
# ============================================================

def validate_dataset(df):

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing telemetry columns:\n"
            + "\n".join(missing)
        )


# ============================================================
# Read latest telemetry
# ============================================================

def get_latest_state():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "Telemetry dataset is empty."
        )

    validate_dataset(df)

    # Always maintain chronological order.
    df = df.sort_values(
        by=["timestamp", "ue_id"],
        kind="stable"
    ).reset_index(drop=True)

    # Latest timestamp in the network.
    latest_timestamp = df["timestamp"].max()

    # Get all UE measurements belonging to
    # the latest timestamp.
    latest_rows = df[
        df["timestamp"] == latest_timestamp
    ].copy()

    latest_rows = latest_rows.sort_values(
        by=["ue_id"],
        kind="stable"
    )

    # Convert records to JSON-compatible dictionaries.
    ue_states = {}

    for _, row in latest_rows.iterrows():

        ue_id = int(row["ue_id"])

        record = {}

        for column in REQUIRED_COLUMNS:

            value = row[column]

            if pd.isna(value):
                record[column] = None

            elif column in {
                "ue_id",
                "cell_id",
                "ue_count",
                "gnb_count",
                "tx_packets",
                "rx_packets",
                "tx_bytes",
                "rx_bytes",
                "cqi",
                "mcs",
                "ri",
            }:
                record[column] = int(value)

            else:
                record[column] = float(value)

        ue_states[str(ue_id)] = record

    # Network state
    first_row = latest_rows.iloc[0]

    state = {
        "synchronization_timestamp": float(latest_timestamp),

        "network": {
            "ue_count": int(first_row["ue_count"]),
            "gnb_count": int(first_row["gnb_count"]),
        },

        "ue_states": ue_states,
    }

    return state

# ============================================================
# Atomic JSON write
# ============================================================

def save_state(state):

    temporary_file = STATE_FILE.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            indent=4
        )

        file.write("\n")

    # Replace old state only after the new file
    # has been completely written.
    os.replace(
        temporary_file,
        STATE_FILE
    )

# ============================================================
# Synchronize once
# ============================================================

def synchronize():

    state = get_latest_state()

    save_state(state)

    timestamp = state[
        "synchronization_timestamp"
    ]

    ue_count = len(
        state["ue_states"]
    )

    print(
        f"[SYNC] timestamp={timestamp:.3f} "
        f"UEs={ue_count} "
        f"-> {STATE_FILE}"
    )


# ============================================================
# Continuous synchronization
# ============================================================

def main():

    print("==============================================")
    print(" NDT MODULE 3")
    print(" Continuous Synchronization Service")
    print("==============================================")

    print(f"Input : {INPUT_FILE}")
    print(f"State : {STATE_FILE}")
    print(f"Interval : {SYNC_INTERVAL} second(s)")
    print()

    last_modified = None

    try:

        while True:

            if INPUT_FILE.exists():

                current_modified = (
                    INPUT_FILE.stat().st_mtime_ns
                )

                # Only synchronize when the dataset
                # has changed.
                if current_modified != last_modified:

                    try:

                        synchronize()

                        last_modified = (
                            current_modified
                        )

                    except Exception as error:

                        print(
                            f"[SYNC ERROR] {error}"
                        )

            else:

                print(
                    "[SYNC] Waiting for telemetry file..."
                )

            time.sleep(SYNC_INTERVAL)

    except KeyboardInterrupt:

        print()
        print("[SYNC] Service stopped.")
        print("[SYNC] Current DT state preserved.")


# ============================================================
# Program entry
# ============================================================

if __name__ == "__main__":
    main()

