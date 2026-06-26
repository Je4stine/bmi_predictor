"""
Step 6 — Train first on VisualBodyToBMI only.

Output: runs/step6_visualbmi/ (sinbmi_best.keras, history.csv)
"""

import argparse

import numpy as np
import tensorflow as tf

import sinbmi_lib as lib
import sinbmi_model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40,
                        help="Phase 2 (end-to-end fine-tune) epochs.")
    parser.add_argument("--head_epochs", type=int, default=15,
                        help="Phase 1 (frozen backbone, head only) epochs.")
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--limit", type=int, default=0, help="Smoke testing.")
    parser.add_argument("--output_dir", default=str(lib.HERE / "runs" / "step6_visualbmi"))
    args = parser.parse_args()

    tf.random.set_seed(lib.SEED)
    np.random.seed(lib.SEED)

    train_df = M.load_split("train")
    val_df = M.load_split("val")
    train_df = train_df[train_df["source"] == "visual_bmi"].reset_index(drop=True)
    val_df = val_df[val_df["source"] == "visual_bmi"].reset_index(drop=True)
    if args.limit:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(max(args.limit // 4, M.BATCH_SIZE))
    print(f"Step 6 (visual_bmi only): train={len(train_df)} val={len(val_df)}")

    model, base = M.build_sinbmi()
    best, ckpt = M.fit_two_phase(model, base,
                                 lambda: M.make_dataset(train_df, True),
                                 lambda: M.make_dataset(val_df, False),
                                 args.output_dir, head_epochs=args.head_epochs,
                                 epochs=args.epochs, head_lr=args.head_lr,
                                 ft_lr=args.lr)

    results = best.evaluate(M.make_dataset(val_df, False), return_dict=True, verbose=0)
    print(f"Step 6 best on visual_bmi val: MAE={results['mae']:.3f} "
          f"MAPE={results['mape']:.2f}%")
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
