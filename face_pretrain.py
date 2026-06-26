"""
Pretrain a face -> BMI model on the Kaggle face-to-bmi-vit dataset.

The output is a backbone (EfficientNetV2-B0 by default) whose weights have
already learned BMI-relevant face features. We will use those weights to
initialize the face branch of the two-branch (body + face) BMI model trained
on Visual BMI.

Usage:
    python face_pretrain.py \\
        --data_dir face-to-bmi-vit-main/data \\
        --output_dir outputs_face

The script writes:
    outputs_face/best_bmi_model.keras   - full pretrained face model
    outputs_face/face_backbone.weights.h5 - just the backbone weights
    outputs_face/{train,val,test}_split.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

import bmi_model


IMG_SIZE = bmi_model.IMG_SIZE
BATCH_SIZE = bmi_model.BATCH_SIZE
SEED = bmi_model.SEED

BMI_BINS = [0, 18.5, 25, 30, 35, 40, 100]


def load_face_dataframe(data_dir, min_side=96):
    """Load the face-to-bmi CSV, filter to rows with usable images."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "data.csv"
    images_dir = data_dir / "Images"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"CSV rows: {len(df)}")

    df = df[df["bmi"].notna()].copy()
    df["target_bmi"] = df["bmi"].astype("float32")
    df = df[(df["target_bmi"] >= 10) & (df["target_bmi"] <= 60)]

    if "gender" in df.columns:
        df["aux_is_female"] = df["gender"].astype(str).str.strip().str.lower().map(
            {"female": 1.0, "male": 0.0}
        ).astype("float32")
    else:
        df["aux_is_female"] = np.nan

    df["image_path_resolved"] = df["name"].apply(lambda n: str(images_dir / n))
    df = df[df["image_path_resolved"].apply(os.path.exists)]
    print(f"Rows with image present: {len(df)}")

    def big_enough(path):
        try:
            with Image.open(path) as im:
                w, h = im.size
            return min(w, h) >= min_side
        except Exception:
            return False

    df = df[df["image_path_resolved"].apply(big_enough)]
    print(f"Rows with min side >= {min_side}px: {len(df)}")

    return df.reset_index(drop=True)


def stratified_bmi_split(df, val_frac=0.10, test_frac=0.15):
    """No person IDs in this dataset, so just stratify by BMI bin."""
    bins = pd.cut(df["target_bmi"], bins=BMI_BINS, include_lowest=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=val_frac + test_frac,
        random_state=SEED,
        stratify=bins,
    )

    temp_bins = pd.cut(
        temp_df["target_bmi"], bins=BMI_BINS, include_lowest=True
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac / (val_frac + test_frac),
        random_state=SEED,
        stratify=temp_bins,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_dataset(dataframe, training=True):
    paths = dataframe["image_path_resolved"].astype(str).values
    labels = dataframe["target_bmi"].astype("float32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(bmi_model.load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000, seed=SEED)
        ds = ds.map(bmi_model.augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def find_backbone_layer(model):
    """Locate the nested backbone sub-model inside our composed model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise RuntimeError("Could not find a nested backbone sub-model.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="face-to-bmi-vit-main/data",
        help="Folder containing data.csv and Images/",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_face",
        help="Where to write the pretrained face model and backbone weights",
    )
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=15)
    parser.add_argument(
        "--backbone",
        default="efficientnetv2b0",
        choices=list(bmi_model.BACKBONES),
    )
    parser.add_argument(
        "--min_side",
        type=int,
        default=96,
        help="Drop images smaller than this on the shorter side",
    )

    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_face_dataframe(args.data_dir, min_side=args.min_side)

    train_df, val_df, test_df = stratified_bmi_split(df)
    print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    # Pretrain face -> BMI as a single-input model. We deliberately do not feed
    # `is_female` into the face pretrain head: we want the backbone features
    # themselves to encode gender-relevant cues, since the joint two-branch
    # model will get gender via its own concatenated aux input later.
    model, base_model = bmi_model.build_model(
        backbone_name=args.backbone,
        use_aux=False,
    )

    print("\nFace pretrain model summary:\n")
    model.summary()

    best_model, checkpoint_path = bmi_model.train_model(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=output_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )

    bmi_model.evaluate_model(best_model, test_ds)

    backbone_layer = find_backbone_layer(best_model)
    backbone_weights_path = output_dir / "face_backbone.weights.h5"
    backbone_layer.save_weights(str(backbone_weights_path))
    print(f"\nSaved face backbone weights to: {backbone_weights_path}")
    print(f"Backbone layer name: {backbone_layer.name}")
    print(f"Best face model:     {checkpoint_path}")


if __name__ == "__main__":
    main()
