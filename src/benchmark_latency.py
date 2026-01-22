# src/benchmark_latency.py
from __future__ import annotations
import os
import time
import statistics
import csv
import datetime as dt
import torch
import torch.nn as nn
from torchvision import models
from src.pt_data_loader import get_loaders_stratified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/pt_resnet18_best.pth"
IMG_SIZE = 224
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
DATASET_PATH = "data/APPLE_DISEASE_DATASET"
OUT_DIR = "reports/metrics"
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval().to(DEVICE)
    return m


def time_forward(model: nn.Module, x: torch.Tensor) -> float:
    # returns elapsed milliseconds for a single forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def main():
    # build val loader so we can also time on real images
    _, val_loader, class_names = get_loaders_stratified(
        dataset_path=DATASET_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED
    )
    model = load_model(len(class_names))

    # 1) random tensor latency (model-only)
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for _ in range(10):  # warmup
        _ = time_forward(model, dummy)

    N = 200
    times_ms = [time_forward(model, dummy) for _ in range(N)]

    # 2) real validation images (batch forward → per-image)
    real_times = []
    counted = 0
    MAX_IMGS = 200
    for xb, _ in val_loader:
        xb = xb.to(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(xb)
        t1 = time.perf_counter()
        per_img_ms = (t1 - t0) * 1000.0 / xb.size(0)
        real_times.extend([per_img_ms] * xb.size(0))
        counted += xb.size(0)
        if counted >= MAX_IMGS:
            break
    real_times = real_times[:MAX_IMGS]

    def stats(arr):
        arr_sorted = sorted(arr)
        return dict(
            mean_ms=float(statistics.mean(arr)),
            std_ms=float(statistics.pstdev(arr) if len(arr) > 1 else 0.0),
            p50_ms=float(statistics.median(arr)),
            p90_ms=float(arr_sorted[int(0.90 * (len(arr_sorted)-1))]),
            p95_ms=float(arr_sorted[int(0.95 * (len(arr_sorted)-1))]),
            n=len(arr)
        )

    s_rand = stats(times_ms)
    s_real = stats(real_times)
    ts = dt.datetime.now().isoformat(timespec="seconds")

    rows = [
        dict(mode="local_random", device=str(DEVICE), timestamp=ts, **s_rand),
        dict(mode="local_real_val", device=str(
            DEVICE), timestamp=ts, **s_real),
    ]
    out_csv = os.path.join(OUT_DIR, "latency_local.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Local model latency written to:", out_csv)
    print("local_random:", s_rand)
    print("local_real_val:", s_real)


if __name__ == "__main__":
    main()
