"""
Step 7 — Train again on the combined dataset (VisualBodyToBMI + 2DImage2BMI),
initialized from the step 6 checkpoint, and export the TFLite model.

Output: runs/step7_combined/ (sinbmi_best.keras, sinbmi_body.tflite)
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

import sinbmi_lib as lib
import sinbmi_model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--init_from",
                        default=str(lib.HERE / "runs" / "step6_visualbmi" / "sinbmi_best.keras"))
    parser.add_argument("--limit", type=int, default=0, help="Smoke testing.")
    parser.add_argument("--output_dir", default=str(lib.HERE / "runs" / "step7_combined"))
    args = parser.parse_args()

    tf.random.set_seed(lib.SEED)
    np.random.seed(lib.SEED)

    train_df = M.load_split("train")
    val_df = M.load_split("val")
    if args.limit:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(max(args.limit // 4, M.BATCH_SIZE))
    print(f"Step 7 (combined): train={len(train_df)} val={len(val_df)}")
    print(train_df["source"].value_counts().to_string())

    ckpt_path = M.resolve_checkpoint(args.init_from)
    if ckpt_path is not None:
        print(f"Initializing from step 6 checkpoint: {ckpt_path}")
        model = tf.keras.models.load_model(ckpt_path)
    else:
        print("WARNING: step 6 checkpoint not found — training from ImageNet init.")
        model, _ = M.build_sinbmi()

    M.compile_model(model, lr=args.lr)
    best, ckpt = M.fit(model, M.make_dataset(train_df, True),
                       M.make_dataset(val_df, False),
                       args.output_dir, epochs=args.epochs)

    results = best.evaluate(M.make_dataset(val_df, False), return_dict=True, verbose=0)
    print(f"Step 7 best on combined val: MAE={results['mae']:.3f} "
          f"MAPE={results['mape']:.2f}%")

    M.export_tflite(best, args.output_dir)
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
