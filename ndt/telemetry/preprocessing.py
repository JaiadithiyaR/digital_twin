import os
import pandas as pd
base_dir = os.path.expanduser("~/NDT")
input_file = os.path.join(base_dir,"ndt","data","telemetry.csv")
output_file = os.path.join(base_dir,"ndt","data","telemetry_cleaned.csv")
df=pd.read_csv(input_file)
columns = [
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
print(f"input: f{input_file}")
if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Dataset not found: {input_file}"
    )
df = pd.read_csv(input_file)
print(f"original shape: {df.shape}")

missing_columns = [
    c for c in columns if c not in df.columns
]
if missing_columns:
    raise ValueError(
        "Missing required columns:\n"+"\n".join(missing_columns)
    )

df = df[columns].copy()

for c in columns:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )
before = len(df)
df = df.dropna(
    subset=["timestamp", "ue_id"]
)
print(
    f"Removed invalid timestamp/UE rows: "
    f"{before - len(df)}"
)

df = df.sort_values(
    by=["timestamp","ue_id"],
    kind="stable"
).reset_index(drop=True)

before = len(df)
df = df.drop_duplicates(
    subset=["timestamp","ue_id"],
    keep="first"
).reset_index(drop=True)

print(
    f"Removed duplicate timestamp/UE rows: "
    f"{before-len(df)}"
)

numeric_columns = [
    c for c in columns if c not in ["timestamp","ue_id"] 
]
df[numeric_columns] = (
    df.groupby("ue_id",group_keys=False)[numeric_columns].ffill()
)

for column in numeric_columns:
    if df[column].isna().any():
        median_value = df[column].median()

        if pd.notna(median_value):
            df[column] = df[column].fillna(
                median_value
            )

before = len(df)
df = df.dropna().reset_index(drop=True)

print(
    f"Removed remaining invalid rows: "f"{before - len(df)}"
)

non_negative_columns = [
    "ue_count",
    "gnb_count",
    "ue_speed_mps",
    "throughput_mbps",
    "offered_load_mbps",
    "latency_ms",
    "jitter_ms",
    "tx_packets",
    "rx_packets",
    "tx_bytes",
    "rx_bytes",
    "cqi",
    "mcs",
    "ri",
    "scheduled_ues",
    "ue_tx_power_dbm",
]

for c in non_negative_columns:
    df[c] = df[c].clip(lower=0)

percentage_columns = [
    "packet_loss_percent",
    "packet_delivery_ratio_percent",
    "prb_utilization_percent",
    "symbol_utilization_percent",
]

for c in percentage_columns:
    df[c] = df[c].clip(
        lower=0,
        upper=100
    )
df = df.sort_values(
    by=["timestamp","ue_id"],
    kind="stable"
).reset_index(drop=True)

df.to_csv(
    output_file,index=False,float_format="%.6f"
)

print("\n========================================")
print(" PREPROCESSING COMPLETED")
print("========================================")

print(f"Original rows : {len(pd.read_csv(input_file))}")
print(f"Final rows    : {len(df)}")
print(f"Columns       : {len(df.columns)}")

print("\nUE distribution:")
print(df["ue_id"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nTime range:")
print(f"Start: {df['timestamp'].min()}")
print(f"End  : {df['timestamp'].max()}")

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nOutput:")
print(output_file)
