"""
Step 8 — Evaluate separately on both datasets.

Reports MAE / MAPE / RMSE on the held-out test split: per dataset, and per
BMI category (underweight / normal / overweight / obese / extremely obese),
mirroring the SinBMI paper's evaluation.

Output: runs/<run>/test_predictions.csv + printed tables.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import sinbmi_lib as lib
import sinbmi_model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=str(lib.HERE / "runs" / "step7_combined" / "sinbmi_best.keras"))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    test_df = M.load_split(args.split)
    print(f"Evaluating {args.model} on {args.split} split ({len(test_df)} rows)\n")

    preds = model.predict(M.make_dataset(test_df, training=False), verbose=0).flatten()
    test_df = test_df.copy()
    test_df["pred_bmi"] = preds
    test_df["category"] = test_df["bmi"].apply(lib.category_of)

    print("Per dataset:")
    rows = [lib.report_metrics(test_df["bmi"], test_df["pred_bmi"], "combined")]
    for source in sorted(test_df["source"].unique()):
        part = test_df[test_df["source"] == source]
        rows.append(lib.report_metrics(part["bmi"], part["pred_bmi"], source))

    print("\nPer BMI category (combined):")
    for name, lo, hi in lib.BMI_CATEGORIES:
        part = test_df[test_df["category"] == name]
        if len(part):
            rows.append(lib.report_metrics(part["bmi"], part["pred_bmi"], name))

    out_dir = Path(args.model).parent
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "test_metrics.csv", index=False)
    print(f"\nSaved predictions + metrics to {out_dir}")


if __name__ == "__main__":
    main()
