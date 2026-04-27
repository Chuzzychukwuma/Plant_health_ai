from src.pt_data_loader import get_loaders_stratified
from src.pt_model import load_model
import time
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


device = torch.device("cpu")

model = load_model()
model.eval()
model.to(device)

_, val_loader = get_loaders_stratified(batch_size=1)

times = []

with torch.no_grad():
    for i, (x, _) in enumerate(val_loader):
        if i == 0:
            model(x.to(device))  # warm-up
            continue

        start = time.perf_counter()
        model(x.to(device))
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

        if i == 100:
            break

print(f"Mean latency: {sum(times)/len(times):.2f} ms")
print(f"P50 latency: {sorted(times)[len(times)//2]:.2f} ms")
