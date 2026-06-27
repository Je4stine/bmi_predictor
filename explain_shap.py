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
    python explain_shap.py --branch body --body-image path/to/body.jpg --face-image path/to/face.jpg --is-female 1
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
PROJECT_ROOT = Path(__file__).resolve().parent


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
    parser.add_argument(
        "--body-image",
        default=None,
        help="Path to a specific body image to explain instead of reading split CSV rows.",
    )
    parser.add_argument(
        "--face-image",
        default=None,
        help="Path to the paired face image when using --body-image.",
    )
    parser.add_argument(
        "--is-female",
        type=float,
        choices=(0.0, 1.0),
        default=None,
        help="Auxiliary model input for direct image mode: 1 for female, 0 for not female.",
    )
    parser.add_argument(
        "--actual-bmi",
        type=float,
        default=np.nan,
        help="Optional actual BMI for direct image mode. Used only in titles and summary CSV.",
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


def resolve_image_path(path, branch):
    path = Path(str(path))
    if path.exists():
        return path

    parts = list(path.parts)
    branch_dir = "body" if branch == "body" else "face"
    candidates = []

    if branch_dir in parts:
        idx = parts.index(branch_dir)
        suffix = Path(*parts[idx + 1 :])
        candidates.extend(
            [
                PROJECT_ROOT / "visual_bmi_cropped" / branch_dir / suffix,
                PROJECT_ROOT / "visual_bmi_cropped" / branch_dir / "Visual BMI" / suffix,
            ]
        )

    basename = path.name
    candidates.extend(PROJECT_ROOT.glob(f"visual_bmi_cropped/{branch_dir}/**/{basename}"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {branch} image. Original path: {path}"
    )


def load_image(path, branch):
    image_path = resolve_image_path(path, branch)
    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(image, dtype=np.float32) / 255.0


def predict_bmi(model, body, face, is_female):
    return float(model.predict([body, face, is_female], verbose=0).reshape(-1)[0])


def make_direct_image_row(args):
    if args.body_image is None and args.face_image is None:
        return None

    missing = []
    if args.body_image is None:
        missing.append("--body-image")
    if args.face_image is None:
        missing.append("--face-image")
    if args.is_female is None:
        missing.append("--is-female")
    if missing:
        raise ValueError(
            "Direct image mode requires " + ", ".join(missing)
        )

    body_path = Path(args.body_image).expanduser()
    face_path = Path(args.face_image).expanduser()
    if not body_path.exists():
        raise FileNotFoundError(f"Body image not found: {body_path}")
    if not face_path.exists():
        raise FileNotFoundError(f"Face image not found: {face_path}")

    return {
        "body_path_resolved": str(body_path),
        "face_path_resolved": str(face_path),
        "aux_is_female": args.is_female,
        "target_bmi": args.actual_bmi,
        "image_id": body_path.stem,
    }


def explain_one_row(shap, model, row, row_index, branch, max_evals, mask, output_dir):
    body_path = resolve_image_path(row["body_path_resolved"], "body")
    face_path = resolve_image_path(row["face_path_resolved"], "face")
    body = load_image(body_path, "body")[None, ...]
    face = load_image(face_path, "face")[None, ...]
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
        "body_path": str(body_path),
        "face_path": str(face_path),
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

    direct_row = make_direct_image_row(args)
    if direct_row is None:
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
        rows_to_explain = [(row_index, df.iloc[row_index]) for row_index in range(start, end)]
    else:
        rows_to_explain = [("custom", direct_row)]

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    summary_rows = []
    for row_index, row in rows_to_explain:
        print(f"Explaining {args.branch} branch for row {row_index}...")
        summary_rows.append(
            explain_one_row(
                shap=shap,
                model=model,
                row=row,
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
