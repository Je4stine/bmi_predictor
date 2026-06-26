"""
Step 2 — Clean 2DImage2BMI metadata and images.

Labels are parsed from filenames ({pid}_{M|F}_{age}_{height*1e5}_{weight*1e5});
the body box is computed from the released segmentation masks. Images stay in
place under the dataset root — only the small index CSV is written.

Output: work/image2bmi_clean.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sinbmi_lib as lib

SPLIT_DIRS = ["Image_train", "Image_val", "Image_test"]
MASK_BODY_THRESHOLD = 200  # mask: body ~96-128 gray (+ black dots), bg ~224+
BODY_PAD_FRAC = 0.04


def parse_filename(name):
    stem = re.sub(r"\s*\(\d+\)$", "", Path(name).stem)  # "... (2)" duplicates
    parts = [p for p in stem.split("_") if p]           # trailing underscores
    if len(parts) != 5:
        return None
    pid, sex, age, height_raw, weight_raw = parts
    sex = sex.upper()
    if sex not in ("M", "F"):
        return None
    try:
        height_m = int(height_raw) / 1e5
        weight_kg = int(weight_raw) / 1e5
        age = int(age)
    except ValueError:
        return None
    if height_m <= 0:
        return None
    return {
        "person_id": f"i2b_{pid}",
        "is_female": 1.0 if sex == "F" else 0.0,
        "age": age,
        "height_m": round(height_m, 5),
        "weight_kg": round(weight_kg, 5),
        "bmi": round(weight_kg / height_m ** 2, 4),
    }


def body_box_from_mask(mask_path):
    try:
        with Image.open(mask_path) as im:
            mask = np.array(im.convert("L"))
    except Exception:
        return None
    body = mask < MASK_BODY_THRESHOLD
    if body.sum() < 500:
        return None
    h, w = mask.shape
    ys, xs = np.where(body)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y1 - y0 < 32 or x1 - x0 < 32:
        return None
    pad_y = BODY_PAD_FRAC * (y1 - y0)
    pad_x = BODY_PAD_FRAC * (x1 - x0)
    return (
        round(max(0.0, (y0 - pad_y) / h), 5),
        round(max(0.0, (x0 - pad_x) / w), 5),
        round(min(1.0, (y1 + pad_y) / h), 5),
        round(min(1.0, (x1 + pad_x) / w), 5),
    )


def main():
    config = lib.load_config()
    root = Path(config["image2bmi_root"])

    rows = []
    n_bad_name = n_no_mask = 0
    for dirname in SPLIT_DIRS:
        img_dir = root / dirname
        mask_dir = root / f"{dirname}_mask"
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            labels = parse_filename(img_path.name)
            if labels is None:
                n_bad_name += 1
                continue
            mask_path = mask_dir / f"Mask_{img_path.name}"
            box = body_box_from_mask(mask_path) if mask_path.exists() else None
            if box is None:
                n_no_mask += 1
                box = (0.0, 0.0, 1.0, 1.0)
            rows.append({
                "source": "image2bmi",
                "image_path": str(img_path),
                "y0": box[0], "x0": box[1], "y1": box[2], "x1": box[3],
                **labels,
            })

    print(f"Parsed {len(rows)} images "
          f"(bad filenames: {n_bad_name}, mask fallbacks: {n_no_mask})")

    out = pd.DataFrame(rows)[lib.COMMON_COLUMNS]
    out = lib.clean_common(out, name="image2bmi")

    out_dir = lib.HERE / "work"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "image2bmi_clean.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows, "
          f"{out['person_id'].nunique()} subjects)")


if __name__ == "__main__":
    main()
