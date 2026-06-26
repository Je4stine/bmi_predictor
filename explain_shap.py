"""
Generate SHAP image explanations for the outputs_v4 two-branch BMI model.

The outputs_v4 model has three inputs:
  - body image, RGB float32 in 0..1, shape [224, 224, 3]
  - face image, RGB float32 in 0..1, shape [224, 224, 3]
  - is_female scalar, shape [1]

This script explains one image branch at a time. For example, when explaining
the body branch, the paired face crop and is_female value are held fixed.

Example:
    python explain_shap.py --branch body --row-index 0
    python explain_shap.py --branch face --row-index 0 --max-evals 800
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMG_SIZE = 224
DEFAULT_OUTPUT_DIR = Path("outputs_v4/shap_explanations")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create SHAP heatmaps for outputs_v4/best_two_branch.keras."
    )
    parser.add_argument(
        "--model",
        default="outputs_v4/best_two_branch.keras",
        help="Path to the Keras model to explain.",
    )
    parser.add_argument(
        "--split-csv",
        default="outputs_v4/test_split.csv",
        help="CSV containing body_path_resolved, face_path_resolved, and aux_is_female.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where SHAP PNGs and summary CSV will be written.",
    )
    parser.add_argument(
        "--branch",
        choices=("body", "face"),
        default="body",
        help="Which image input to explain.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="First row in the split CSV to explain.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of consecutive rows to explain.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=500,
        help="SHAP evaluation budget. Higher is slower and usually smoother.",
    )
    parser.add_argument(
        "--mask",
        default="blur(32,32)",
        help='Image masking strategy, such as "blur(32,32)", "inpaint_telea", or "inpaint_ns".',
    )
    return parser.parse_args()


def require_runtime_imports():
    try:
        import shap
        import tensorflow as tf
    except ImportError as exc:
        print(
            "Missing runtime dependency. Install SHAP and make sure TensorFlow "
            "imports cleanly in the active environment:\n"
            "  python -m pip install shap\n"
            f"\nOriginal error: {exc}",
            file=sys.stderr,
        )
        raise
    return shap, tf


def load_image(path):
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(image, dtype=np.float32) / 255.0


def predict_bmi(model, body, face, is_female):
    return float(model.predict([body, face, is_female], verbose=0).reshape(-1)[0])


def explain_one_row(shap, model, row, row_index, branch, max_evals, mask, output_dir):
    body = load_image(row["body_path_resolved"])[None, ...]
    face = load_image(row["face_path_resolved"])[None, ...]
    is_female = np.asarray([[float(row["aux_is_female"])]], dtype=np.float32)

    actual_bmi = float(row["target_bmi"])
    predicted_bmi = predict_bmi(model, body, face, is_female)

    if branch == "body":
        image_to_explain = body

        def predict_from_image(x):
            repeated_face = np.repeat(face, x.shape[0], axis=0)
            repeated_aux = np.repeat(is_female, x.shape[0], axis=0)
            return model.predict([x, repeated_face, repeated_aux], verbose=0)

    else:
        image_to_explain = face

        def predict_from_image(x):
            repeated_body = np.repeat(body, x.shape[0], axis=0)
            repeated_aux = np.repeat(is_female, x.shape[0], axis=0)
            return model.predict([repeated_body, x, repeated_aux], verbose=0)

    masker = shap.maskers.Image(mask, image_to_explain[0].shape)
    explainer = shap.Explainer(
        predict_from_image,
        masker,
        algorithm="partition",
        output_names=["predicted_bmi"],
    )

    explanation = explainer(image_to_explain, max_evals=max_evals)

    title = (
        f"{branch} row {row_index} | "
        f"predicted BMI {predicted_bmi:.2f} | actual BMI {actual_bmi:.2f}"
    )
    shap.plots.image(explanation, pixel_values=image_to_explain, show=False)
    plt.suptitle(title, y=1.02)

    image_id = str(row.get("image_id", row_index)).replace("/", "_")
    output_path = output_dir / f"shap_{branch}_row_{row_index}_{image_id}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close("all")

    return {
        "row_index": row_index,
        "branch": branch,
        "image_id": row.get("image_id", ""),
        "body_path": row["body_path_resolved"],
        "face_path": row["face_path_resolved"],
        "is_female": float(row["aux_is_female"]),
        "actual_bmi": actual_bmi,
        "predicted_bmi": predicted_bmi,
        "output_path": str(output_path),
    }


def main():
    args = parse_args()
    shap, tf = require_runtime_imports()

    model_path = Path(args.model)
    split_csv = Path(args.split_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    required = {
        "body_path_resolved",
        "face_path_resolved",
        "aux_is_female",
        "target_bmi",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {split_csv}: {missing}")

    start = args.row_index
    end = start + args.num_samples
    if start < 0 or end > len(df):
        raise IndexError(
            f"Requested rows [{start}, {end}) but {split_csv} has {len(df)} rows."
        )

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    summary_rows = []
    for row_index in range(start, end):
        print(f"Explaining {args.branch} branch for row {row_index}...")
        summary_rows.append(
            explain_one_row(
                shap=shap,
                model=model,
                row=df.iloc[row_index],
                row_index=row_index,
                branch=args.branch,
                max_evals=args.max_evals,
                mask=args.mask,
                output_dir=output_dir,
            )
        )

    summary_path = output_dir / f"summary_{args.branch}.csv"
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done. Wrote {len(summary_rows)} explanation(s) to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
