"""
Step 10 — Fine-tune on your own images.

Starts from the step 7 checkpoint, fine-tunes at a low learning rate on your
own app-style photos (work/own_clean.csv from step 9) mixed with a slice of
the combined training data (so the model doesn't forget the base task), and
exports the final TFLite model.

Output: runs/step10_own/ (sinbmi_best.keras, sinbmi_body.tflite)
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--replay_frac", type=float, default=0.25,
                        help="Fraction of base training data mixed in.")
    parser.add_argument("--init_from",
                        default=str(lib.HERE / "runs" / "step7_combined" / "sinbmi_best.keras"))
    parser.add_argument("--output_dir", default=str(lib.HERE / "runs" / "step10_own"))
    args = parser.parse_args()

    tf.random.set_seed(lib.SEED)
    np.random.seed(lib.SEED)

    own_path = lib.HERE / "work" / "own_clean.csv"
    if not own_path.exists():
        raise SystemExit("work/own_clean.csv missing — run step 9 first.")
    own = pd.read_csv(own_path)
    if len(own) < 8:
        raise SystemExit(f"Only {len(own)} own images — collect more first.")

    own = lib.split_by_subject(own, train=0.8, val=0.2)
    own_train = own[own["split"] == "train"]
    own_val = own[own["split"].isin(["val", "test"])]

    base_train = M.load_split("train")
    replay = base_train.sample(
        max(int(len(base_train) * args.replay_frac), len(own_train)),
        random_state=lib.SEED,
    )
    train_df = pd.concat([own_train, replay], ignore_index=True)
    print(f"Step 10: own train={len(own_train)} own val={len(own_val)} "
          f"replay={len(replay)}")

    ckpt_path = M.resolve_checkpoint(args.init_from)
    if ckpt_path is None:
        raise SystemExit(f"Checkpoint not found: {args.init_from} — run step 7 first.")
    model = tf.keras.models.load_model(ckpt_path)
    M.compile_model(model, lr=args.lr)
    best, ckpt = M.fit(model, M.make_dataset(train_df, True),
                       M.make_dataset(own_val, False),
                       args.output_dir, epochs=args.epochs, patience=6)

    preds = best.predict(M.make_dataset(own_val, False), verbose=0).flatten()
    lib.report_metrics(own_val["bmi"], preds, "own (held-out)")

    M.export_tflite(best, args.output_dir)
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
