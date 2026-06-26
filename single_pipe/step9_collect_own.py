"""
Step 9 — Collect your own app-style images.

Drop photos into own_images/ (path configurable in datasets.json) and list
them in own_images/labels.csv with columns:

    filename,weight_kg,height_m,person_id
    me_front_1.jpg,82.5,1.78,me

Photos should match how the app captures them: full body visible, frontal,
roughly 1-3 m from the camera. One subject can have many photos (person_id
keeps them in the same split). This script validates the labels, computes
the body box with MediaPipe Pose (same detector the original pipeline used),
and writes work/own_clean.csv in the common format for step 10.

Output: work/own_clean.csv
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sinbmi_lib as lib

sys.path.insert(0, str(lib.HERE.parent))
import body_crop  # noqa: E402  (MediaPipe pose helpers from the parent repo)


def body_box_from_pose(detector, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    landmarks = body_crop.detect_landmarks(detector, image)
    if landmarks is None:
        return None
    h, w = image.shape[:2]
    xs, ys = body_crop._visible_xy(
        landmarks, list(range(len(landmarks))), image.shape, 0.3
    )
    if len(xs) < 6:
        return None
    pad_x, pad_y = 0.06 * (max(xs) - min(xs)), 0.06 * (max(ys) - min(ys))
    return (
        round(max(0.0, (min(ys) - pad_y) / h), 5),
        round(max(0.0, (min(xs) - pad_x) / w), 5),
        round(min(1.0, (max(ys) + pad_y) / h), 5),
        round(min(1.0, (max(xs) + pad_x) / w), 5),
    )


def main():
    config = lib.load_config()
    own_root = Path(config["own_images_root"])
    labels_path = own_root / "labels.csv"

    if not labels_path.exists():
        own_root.mkdir(parents=True, exist_ok=True)
        labels_path.write_text("filename,weight_kg,height_m,person_id\n")
        print(f"Created {own_root} and a labels.csv template.")
        print("Add photos + label rows there, then re-run this step.")
        return

    df = pd.read_csv(labels_path)
    if df.empty:
        print(f"{labels_path} has no rows yet — add your photos first.")
        return
    print(f"Label rows: {len(df)}")

    detector = body_crop.make_pose_landmarker()
    rows = []
    try:
        for _, r in df.iterrows():
            path = own_root / str(r["filename"])
            if not path.exists():
                print(f"  missing file, skipped: {path.name}")
                continue
            height_m = float(r["height_m"])
            weight_kg = float(r["weight_kg"])
            box = body_box_from_pose(detector, path)
            if box is None:
                print(f"  no pose found, using full frame: {path.name}")
                box = (0.0, 0.0, 1.0, 1.0)
            rows.append({
                "source": "own",
                "person_id": f"own_{r['person_id']}",
                "image_path": str(path),
                "y0": box[0], "x0": box[1], "y1": box[2], "x1": box[3],
                "bmi": round(weight_kg / height_m ** 2, 4),
                "is_female": np.nan,
                "height_m": height_m,
                "weight_kg": weight_kg,
                "age": np.nan,
            })
    finally:
        detector.close()

    out = pd.DataFrame(rows)[lib.COMMON_COLUMNS]
    out = lib.clean_common(out, name="own")
    out_path = lib.HERE / "work" / "own_clean.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows) — ready for step 10.")


if __name__ == "__main__":
    main()
