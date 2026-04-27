import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "reports/metrics/latency_local.csv"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Use latest run only
latest_ts = df["timestamp"].max()
df = df[df["timestamp"] == latest_ts]

# Split modes
rand = df[df["mode"] == "local_random"]
real = df[df["mode"] == "local_real_val"]

# -------- Figure L1: Model-only latency --------
plt.figure()
plt.bar(
    ["p50", "p90", "p95"],
    [rand["p50_ms"].iloc[0], rand["p90_ms"].iloc[0], rand["p95_ms"].iloc[0]]
)
plt.ylabel("Latency (ms)")
plt.title("Local Model Inference Latency (CPU)")
plt.savefig(f"{OUT_DIR}/figure_L1_local_latency.png", dpi=300)
plt.close()

# -------- Figure L2: Real image latency --------
plt.figure()
plt.bar(
    ["p50", "p90", "p95"],
    [real["p50_ms"].iloc[0], real["p90_ms"].iloc[0], real["p95_ms"].iloc[0]]
)
plt.ylabel("Latency (ms)")
plt.title("Validation Image Inference Latency (CPU)")
plt.savefig(f"{OUT_DIR}/figure_L2_real_latency.png", dpi=300)
plt.close()

# -------- Figure L3: Comparison --------
plt.figure()
plt.bar(
    ["Model-only", "Real images"],
    [rand["mean_ms"].iloc[0], real["mean_ms"].iloc[0]]
)
plt.ylabel("Mean Latency (ms)")
plt.title("Model-only vs Real Image Latency (CPU)")
plt.savefig(f"{OUT_DIR}/figure_L3_latency_comparison.png", dpi=300)
plt.close()

print("Figures saved to:", OUT_DIR)
