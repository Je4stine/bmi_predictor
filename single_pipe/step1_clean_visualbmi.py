"""
Step 1 — Clean VisualBodyToBMI metadata and images.

Reads visual_bmi_annotations.csv and resolves each row to the body-crop file
under visual_bmi_cropped/body (those crops contain the whole body; the box in
the common schema is therefore the full file). Drops rows whose image is
missing, undecodable, duplicated, or whose BMI is implausible.

Output: work/visualbmi_clean.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

import sinbmi_lib as lib


def main():
    config = lib.load_config()
    root = Path(config["visual_bmi_cropped_root"])
    body_root = root / "body"
    csv_path = root / "visual_bmi_annotations.csv"

    df = pd.read_csv(csv_path)
    print(f"Annotation rows: {len(df)}")

    # Annotation paths are Kaggle-style; resolve by basename under body/.
    by_name = {p.name: p for p in body_root.rglob("*.jpg")}
    print(f"Body crop files: {len(by_name)}")
    df["resolved"] = df["image_path"].apply(
        lambda p: str(by_name.get(Path(str(p)).name, ""))
    )
    missing = (df["resolved"] == "").sum()
    if missing:
        print(f"Rows without a body crop file: {missing} (dropped)")
    df = df[df["resolved"] != ""]

    # Sanity: recompute BMI from reported weight/height and prefer it when
    # the stored value disagrees badly (guards against annotation typos).
    weight_kg = pd.to_numeric(df["weight_lb"], errors="coerce") * 0.45359237
    height_m = pd.to_numeric(df["height_in"], errors="coerce") * 0.0254
    bmi_calc = weight_kg / (height_m ** 2)
    bmi_stored = pd.to_numeric(df["BMI"], errors="coerce")
    disagree = (bmi_stored - bmi_calc).abs() > 1.0
    print(f"Rows where stored BMI disagrees with weight/height by >1: {disagree.sum()}")
    bmi = bmi_calc.where(bmi_calc.notna(), bmi_stored)

    is_female = df["is_female"].astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0}
    )

    out = pd.DataFrame({
        "source": "visual_bmi",
        "person_id": "vb_" + df["person_id"].astype(str),
        "image_path": df["resolved"],
        "y0": 0.0, "x0": 0.0, "y1": 1.0, "x1": 1.0,
        "bmi": bmi.round(4),
        "is_female": is_female,
        "height_m": height_m.round(5),
        "weight_kg": weight_kg.round(5),
        "age": np.nan,
    })[lib.COMMON_COLUMNS]

    out = lib.clean_common(out, name="visual_bmi")

    out_dir = lib.HERE / "work"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "visualbmi_clean.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows, "
          f"{out['person_id'].nunique()} subjects)")


if __name__ == "__main__":
    main()
