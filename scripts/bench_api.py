# scripts/bench_api.py
import os
import time
import csv
import glob
import random
import requests
import statistics
import datetime as dt

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
GLOB_PATTERNS = [
    "data/APPLE_DISEASE_DATASET/*/*.jpg",
    "data/APPLE_DISEASE_DATASET/*/*.jpeg",
    "data/APPLE_DISEASE_DATASET/*/*.png",
    "data/APPLE_DISEASE_DATASET/*/*.JPG",
    "data/APPLE_DISEASE_DATASET/*/*.JPEG",
    "data/APPLE_DISEASE_DATASET/*/*.PNG",
]
OUT_CSV = "reports/metrics/latency_api.csv"
N = 50


def pick_images(n):
    paths = []
    for pat in GLOB_PATTERNS:
        paths.extend(glob.glob(pat))
    random.shuffle(paths)
    return paths[:n]


def stats(arr):
    arr_sorted = sorted(arr)
    return dict(
        mean_ms=float(statistics.mean(arr)),
        std_ms=float(statistics.pstdev(arr) if len(arr) > 1 else 0.0),
        p50_ms=float(statistics.median(arr)),
        p90_ms=float(arr_sorted[int(0.90 * (len(arr_sorted)-1))]),
        p95_ms=float(arr_sorted[int(0.95 * (len(arr_sorted)-1))]),
        n=len(arr),
    )


def main():
    os.makedirs("reports/metrics", exist_ok=True)
    imgs = pick_images(N)
    if not imgs:
        raise SystemExit(
            "No images found under data/APPLE_DISEASE_DATASET/*/*")

    times, oks = [], 0
    for p in imgs:
        with open(p, "rb") as fh:
            files = {"file": (os.path.basename(p), fh, "image/jpeg")}
            t0 = time.perf_counter()
            r = requests.post(API_URL, files=files, timeout=30)
            t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        if r.status_code == 200:
            oks += 1

    s = stats(times)
    s.update(dict(success=oks, api_url=API_URL,
             timestamp=dt.datetime.now().isoformat(timespec="seconds")))
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s.keys()))
        if write_header:
            w.writeheader()
        w.writerow(s)

    print("API latency written to:", OUT_CSV)
    print(s)


if __name__ == "__main__":
    main()
