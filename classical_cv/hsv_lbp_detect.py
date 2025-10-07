# classical_cv/hsv_lbp_detect.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_DIR = Path("data/APPLE_DISEASE_DATASET")  # adjust if different


def analyze_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # HSV threshold for yellow/brown discoloration (tune these values)
    lower = np.array([5, 40, 40])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    spot_ratio = float(mask.sum()) / mask.size

    # LBP texture (uniform)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    lbp_var = float(np.var(lbp_hist))

    # Basic rule-based decision
    result = "Possible disease" if (
        spot_ratio > 0.03 or lbp_var > 0.02) else "Likely healthy"

    return {
        "path": str(img_path),
        "spot_ratio": spot_ratio,
        "lbp_var": lbp_var,
        "result": result,
        "mask": mask,
        "img": img
    }


def demo_sample(n_per_class=2):
    classes = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    for cls in classes:
        imgs = sorted(list(cls.glob("*.*")))
        if not imgs:
            continue
        sample = imgs[:n_per_class]
        for p in sample:
            r = analyze_image(p)
            print(cls.name, p.name, "=>", r["result"],
                  f"spots={r['spot_ratio']:.3f}", f"lbp_var={r['lbp_var']:.4f}")
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(r["img"])
            plt.title(cls.name)
            plt.subplot(1, 2, 2)
            plt.imshow(r["mask"], cmap="gray")
            plt.title("Detected mask")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    demo_sample()
