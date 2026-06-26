"""
Shared, TF-free helpers for the SinBMI pipeline (steps 1-5, 9).

Common CSV format (one row per image, produced by step 3, consumed onward):
    source       - 'visual_bmi' | 'image2bmi' | 'own'
    person_id    - subject identifier, prefixed per source (split unit)
    image_path   - absolute path to the ORIGINAL image (never copied)
    y0,x0,y1,x1  - fractional body box in the original (0,0,1,1 = whole image)
    bmi          - target
    is_female    - 1.0/0.0/NaN (metadata only; the model does not use it)
    height_m, weight_kg, age - metadata where known (NaN otherwise)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent

IMG_SIZE = 260  # EfficientNet-B2 native resolution
BMI_MIN, BMI_MAX = 10.0, 60.0
SEED = 42

COMMON_COLUMNS = [
    "source", "person_id", "image_path", "y0", "x0", "y1", "x1",
    "bmi", "is_female", "height_m", "weight_kg", "age",
]

BMI_CATEGORIES = [
    ("underweight", 0.0, 18.5),
    ("normal", 18.5, 25.0),
    ("overweight", 25.0, 30.0),
    ("obese", 30.0, 40.0),
    ("extremely_obese", 40.0, 1000.0),
]


def load_config():
    return json.loads((HERE / "datasets.json").read_text())


def is_decodable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def clean_common(df, validate_images=True, name=""):
    """Shared cleaning: file exists, BMI plausible, image decodable."""
    n0 = len(df)
    df = df[df["image_path"].apply(lambda p: Path(str(p)).exists())]
    df = df[(df["bmi"] >= BMI_MIN) & (df["bmi"] <= BMI_MAX)]
    df = df.drop_duplicates(subset=["image_path"])
    if validate_images:
        ok = df["image_path"].apply(is_decodable)
        bad = (~ok).sum()
        if bad:
            print(f"{name}: dropping {bad} undecodable images")
        df = df[ok]
    print(f"{name}: {n0} -> {len(df)} rows after cleaning")
    return df.reset_index(drop=True)


def crop_pad_resize(image, box, size=IMG_SIZE):
    """
    The single canonical image transform (step 4), used identically at
    training and inference: crop fractional body box -> pad to square with
    black -> resize to (size, size). PIL in, PIL out.
    """
    w, h = image.size
    y0, x0, y1, x1 = box
    crop = image.crop((
        int(round(x0 * w)), int(round(y0 * h)),
        max(int(round(x1 * w)), int(round(x0 * w)) + 1),
        max(int(round(y1 * h)), int(round(y0 * h)) + 1),
    ))
    cw, ch = crop.size
    side = max(cw, ch)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return square.resize((size, size), Image.BILINEAR)


def split_by_subject(df, train=0.70, val=0.15, seed=SEED):
    """Step 5: subject-level split. Every image of a person stays together."""
    rng = np.random.default_rng(seed)
    subjects = np.sort(df["person_id"].unique())
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    assign = {}
    for s in subjects[:n_train]:
        assign[s] = "train"
    for s in subjects[n_train:n_train + n_val]:
        assign[s] = "val"
    for s in subjects[n_train + n_val:]:
        assign[s] = "test"

    df = df.copy()
    df["split"] = df["person_id"].map(assign)
    return df


def category_of(bmi):
    for name, lo, hi in BMI_CATEGORIES:
        if lo <= bmi < hi:
            return name
    return "unknown"


def report_metrics(y_true, y_pred, label):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"  {label:24s} n={len(y_true):5d}  MAE={mae:.3f}  "
          f"MAPE={mape:.2f}%  RMSE={rmse:.3f}")
    return {"label": label, "n": len(y_true), "mae": mae, "mape": mape, "rmse": rmse}
