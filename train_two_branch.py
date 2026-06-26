"""
Train a two-branch BMI model on Visual BMI.

Inputs at inference:
  - body image  (224 x 224 x 3, RGB, 0..1)
  - face image  (224 x 224 x 3, RGB, 0..1)
  - is_female   (scalar; 1.0 female, 0.0 male)

Output: predicted BMI (float).

Each branch is an EfficientNetV2-B0. The face branch is initialized from a
backbone produced by face_pretrain.py. The body branch is initialized from
ImageNet. Pooled features from both branches are concatenated with the
is_female scalar before a small regression head.

Usage:
    python train_two_branch.py \\
        --data_dir visual_bmi_cropped \\
        --face_backbone outputs_face/face_backbone.weights.h5 \\
        --output_dir outputs_v3
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import bmi_model


IMG_SIZE = bmi_model.IMG_SIZE
BATCH_SIZE = bmi_model.BATCH_SIZE
SEED = bmi_model.SEED


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_two_branch_dataframe(data_dir):
    """
    Reads the CSV mirrored at the root of the cropped dataset and resolves
    body + face paths for each row. Drops rows missing either crop.
    """
    data_dir = Path(data_dir).resolve()
    body_root = data_dir / "body"
    face_root = data_dir / "face"

    if not body_root.exists() or not face_root.exists():
        raise FileNotFoundError(
            f"Expected {body_root} and {face_root}. "
            "Did you run body_crop.py first?"
        )

    df = bmi_model.load_and_prepare_dataframe(str(body_root))

    # Body paths are already resolved against body_root. The face crops live
    # at the same relative path under face_root, so we derive them directly
    # rather than re-running expensive recursive lookups.
    body_root_resolved = body_root.resolve()

    def derive_face_path(body_path_str):
        try:
            rel = Path(body_path_str).resolve().relative_to(body_root_resolved)
        except ValueError:
            return None
        candidate = face_root / rel
        return str(candidate) if candidate.exists() else None

    print(f"Resolving face paths for {len(df)} rows...")
    df["body_path_resolved"] = df["image_path_resolved"]
    df["face_path_resolved"] = df["image_path_resolved"].apply(derive_face_path)

    before = len(df)
    df = df[df["face_path_resolved"].notna()].reset_index(drop=True)
    print(f"Rows with both body and face crops: {len(df)} (dropped {before - len(df)})")

    if df["aux_is_female"].notna().sum() == 0:
        raise ValueError(
            "is_female column is missing or all-NaN. "
            "The two-branch model requires gender labels."
        )
    df = df.dropna(subset=["aux_is_female"]).reset_index(drop=True)
    print(f"Rows with gender label: {len(df)}")

    return df


def compute_bmi_sample_weights(bmi_values, num_bins=20, min_weight=0.5, max_weight=5.0):
    """
    Per-sample weights inversely proportional to BMI bin frequency.
    Up-weights tail BMIs (under-/very-overweight) that the model otherwise
    regresses toward the dataset mean. Clipped + normalized so mean weight = 1.
    """
    bmi_values = np.asarray(bmi_values, dtype=np.float64)
    hist, bin_edges = np.histogram(bmi_values, bins=num_bins)
    bin_idx = np.clip(np.digitize(bmi_values, bin_edges[1:-1]), 0, num_bins - 1)
    bin_weight = 1.0 / np.maximum(hist, 1)
    weights = bin_weight[bin_idx]
    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def _load_pair(body_path, face_path, aux, bmi):
    body = bmi_model._decode_image(body_path)
    face = bmi_model._decode_image(face_path)
    aux = tf.reshape(tf.cast(aux, tf.float32), [1])
    bmi = tf.cast(bmi, tf.float32)
    return (body, face, aux), bmi


def _load_pair_weighted(body_path, face_path, aux, bmi, weight):
    inputs, bmi = _load_pair(body_path, face_path, aux, bmi)
    return inputs, bmi, tf.cast(weight, tf.float32)


def _augment_pair(inputs, bmi):
    body, face, aux = inputs

    # Same horizontal flip for both crops to preserve body/face correlation.
    flip = tf.random.uniform([]) < 0.5
    body = tf.cond(flip, lambda: tf.image.flip_left_right(body), lambda: body)
    face = tf.cond(flip, lambda: tf.image.flip_left_right(face), lambda: face)

    # Independent geometry/color jitter is fine — the visual augmentations
    # we apply are mild and don't break body/face correspondence.
    body = bmi_model._AUGMENT_LAYERS(body, training=True)
    face = bmi_model._AUGMENT_LAYERS(face, training=True)

    body = tf.image.random_brightness(body, max_delta=0.15)
    body = tf.image.random_contrast(body, lower=0.85, upper=1.15)
    body = tf.image.random_saturation(body, lower=0.85, upper=1.15)
    body = tf.image.random_hue(body, max_delta=0.03)
    body = tf.clip_by_value(body, 0.0, 1.0)
    body = bmi_model.random_erase(body, p=0.25)

    face = tf.image.random_brightness(face, max_delta=0.15)
    face = tf.image.random_contrast(face, lower=0.85, upper=1.15)
    face = tf.image.random_saturation(face, lower=0.85, upper=1.15)
    face = tf.image.random_hue(face, max_delta=0.03)
    face = tf.clip_by_value(face, 0.0, 1.0)

    return (body, face, aux), bmi


def _augment_pair_weighted(inputs, bmi, weight):
    augmented_inputs, augmented_bmi = _augment_pair(inputs, bmi)
    return augmented_inputs, augmented_bmi, weight


def _load_body(body_path, aux, bmi):
    body = bmi_model._decode_image(body_path)
    aux = tf.reshape(tf.cast(aux, tf.float32), [1])
    bmi = tf.cast(bmi, tf.float32)
    return (body, aux), bmi


def _load_body_weighted(body_path, aux, bmi, weight):
    inputs, bmi = _load_body(body_path, aux, bmi)
    return inputs, bmi, tf.cast(weight, tf.float32)


def _augment_body(inputs, bmi):
    body, aux = inputs

    flip = tf.random.uniform([]) < 0.5
    body = tf.cond(flip, lambda: tf.image.flip_left_right(body), lambda: body)
    body = bmi_model._AUGMENT_LAYERS(body, training=True)
    body = tf.image.random_brightness(body, max_delta=0.15)
    body = tf.image.random_contrast(body, lower=0.85, upper=1.15)
    body = tf.image.random_saturation(body, lower=0.85, upper=1.15)
    body = tf.image.random_hue(body, max_delta=0.03)
    body = tf.clip_by_value(body, 0.0, 1.0)
    body = bmi_model.random_erase(body, p=0.25)

    return (body, aux), bmi


def _augment_body_weighted(inputs, bmi, weight):
    augmented_inputs, augmented_bmi = _augment_body(inputs, bmi)
    return augmented_inputs, augmented_bmi, weight


def make_dataset(df, training=True, sample_weights=None, use_face=True):
    body = df["body_path_resolved"].astype(str).values
    aux = df["aux_is_female"].astype("float32").values
    labels = df["target_bmi"].astype("float32").values

    if use_face:
        face = df["face_path_resolved"].astype(str).values
        load_fn = _load_pair_weighted if sample_weights is not None else _load_pair
        aug_fn = _augment_pair_weighted if sample_weights is not None else _augment_pair

        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float32)
            ds = tf.data.Dataset.from_tensor_slices((body, face, aux, labels, weights))
        else:
            ds = tf.data.Dataset.from_tensor_slices((body, face, aux, labels))
    else:
        load_fn = _load_body_weighted if sample_weights is not None else _load_body
        aug_fn = _augment_body_weighted if sample_weights is not None else _augment_body

        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float32)
            ds = tf.data.Dataset.from_tensor_slices((body, aux, labels, weights))
        else:
            ds = tf.data.Dataset.from_tensor_slices((body, aux, labels))

    ds = ds.map(load_fn, num_parallel_calls=2)

    if training:
        ds = ds.shuffle(256, seed=SEED)
        ds = ds.map(aug_fn, num_parallel_calls=2)

    ds = ds.batch(BATCH_SIZE).prefetch(2)
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_renamed_efficientnet(prefix, weights="imagenet"):
    """
    Build an EfficientNetV2-B0 whose inner layers AND outer model name are all
    prefixed, so two instances can coexist in the same functional graph
    (Keras 3 enforces unique names across nested models).
    """
    src = tf.keras.applications.EfficientNetV2B0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=weights,
        include_preprocessing=True,
    )

    def rename(layer):
        config = layer.get_config()
        config["name"] = f"{prefix}_{config['name']}"
        return layer.__class__.from_config(config)

    inner_cloned = tf.keras.models.clone_model(src, clone_function=rename)
    inner_cloned.set_weights(src.get_weights())

    # Rebuild from config so we can set the outer model name. Keras does not
    # expose a direct setter, but `Model.from_config` honors `name` in config.
    config = inner_cloned.get_config()
    config["name"] = f"{prefix}_efficientnetv2_b0"
    final = tf.keras.Model.from_config(config)
    final.set_weights(inner_cloned.get_weights())
    return final


def _make_branch(name, face_weights_path=None):
    """
    Returns (input_tensor, pooled_features, base_model).
    For the face branch, optionally load pretrained backbone weights.
    """
    base = _build_renamed_efficientnet(prefix=name, weights="imagenet")

    if face_weights_path is not None:
        # `.weights.h5` files load positionally (by graph traversal order),
        # which works even though our cloned backbone has prefixed layer
        # names — the structure is identical to the source.
        base.load_weights(face_weights_path)
        print(f"Loaded face backbone weights from {face_weights_path}")

    base.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name=name)
    x = inp * 255.0  # EfficientNetV2 has built-in rescaling
    x = base(x, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)

    return inp, pooled, base


def build_two_branch_model(face_weights_path=None, huber_delta=3.0, use_face=True):
    body_input, body_features, body_base = _make_branch("body")
    aux_input = tf.keras.Input(shape=(1,), name="is_female")

    if use_face:
        face_input, face_features, face_base = _make_branch(
            "face", face_weights_path=face_weights_path
        )
        feature_tensors = [body_features, face_features, aux_input]
        model_inputs = [body_input, face_input, aux_input]
    else:
        face_base = None
        feature_tensors = [body_features, aux_input]
        model_inputs = [body_input, aux_input]

    x = tf.keras.layers.Concatenate(name="features")(feature_tensors)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="linear", name="bmi")(x)

    model = tf.keras.Model(inputs=model_inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )

    return model, body_base, face_base


def unfreeze_backbone(base_model, last_n=60):
    """Unfreeze the last `last_n` layers but keep BatchNorm in inference mode."""
    base_model.trainable = True
    for layer in base_model.layers[:-last_n]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train(model, body_base, face_base, train_ds, val_ds, output_dir,
          stage1_epochs, stage2_epochs, huber_delta=3.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_two_branch.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_mae",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=10,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            mode="min",
            verbose=1,
        ),
    ]

    print("\nStage 1: Training head with both backbones frozen...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks,
    )

    print("\nStage 2: Fine-tuning last layers of backbone(s)...\n")
    unfreeze_backbone(body_base, last_n=60)
    if face_base is not None:
        unfreeze_backbone(face_base, last_n=60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    return best_model, checkpoint_path


def evaluate(model, test_ds):
    print("\nEvaluating on test set...\n")
    results = model.evaluate(test_ds, return_dict=True)
    print("\nTest results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\nSample predictions:")
    for inputs, labels in test_ds.take(1):
        preds = model.predict(inputs, verbose=0).flatten()
        for actual, predicted in zip(labels.numpy()[:10], preds[:10]):
            print(f"Actual BMI: {actual:.2f} | Predicted BMI: {predicted:.2f}")


def export_tflite(model, output_dir, use_face=True):
    output_dir = Path(output_dir)
    tflite_name = "bmi_two_branch.tflite" if use_face else "bmi_body_only.tflite"
    tflite_path = output_dir / tflite_name

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    buf = converter.convert()
    tflite_path.write_bytes(buf)
    print(f"\nSaved TFLite model to: {tflite_path}")
    return tflite_path


def save_metadata(output_dir, use_face=True):
    output_dir = Path(output_dir)

    if use_face:
        text = f"""
Two-branch BMI model metadata

Backbones:
- Body: EfficientNetV2-B0 (ImageNet init, fine-tuned on Visual BMI)
- Face: EfficientNetV2-B0 (face-BMI pretrained, fine-tuned on Visual BMI)

Inputs (3, identify by tensor rank/shape):
- body image:  shape [1, {IMG_SIZE}, {IMG_SIZE}, 3], float32, range 0..1, RGB
- face image:  shape [1, {IMG_SIZE}, {IMG_SIZE}, 3], float32, range 0..1, RGB
- is_female:   shape [1, 1], float32. 1.0 = female, 0.0 = male.

Output:
- shape [1, 1], float32, predicted BMI

Android preprocessing:
- For each image: resize to {IMG_SIZE}x{IMG_SIZE}, RGB, divide pixel values by 255.0
- Detect face on-device (e.g. ML Kit Face Detector); crop a square around the
  face landmarks with ~40% padding before resizing.
- For body, crop to a tight body bounding box with ~15% padding.
- If face detection fails, the recommended fallback is to skip the face crop
  and pass the body image as the face image — model accuracy degrades
  gracefully but does not crash.

Important:
- Estimated BMI from images. Not a medical diagnosis.
- Accuracy depends on framing, pose, clothing, lighting.
"""
    else:
        text = f"""
Body-only BMI model metadata

Backbone:
- Body: EfficientNetV2-B0 (ImageNet init, fine-tuned on Visual BMI)

Inputs (2, identify by tensor rank/shape):
- body image:  shape [1, {IMG_SIZE}, {IMG_SIZE}, 3], float32, range 0..1, RGB
- is_female:   shape [1, 1], float32. 1.0 = female, 0.0 = male.

Output:
- shape [1, 1], float32, predicted BMI

Android preprocessing:
- Resize the body crop to {IMG_SIZE}x{IMG_SIZE}, RGB, divide pixel values by 255.0
- Crop to a tight body bounding box with ~15% padding.

Important:
- Estimated BMI from images. Not a medical diagnosis.
- Accuracy depends on framing, pose, clothing, lighting.
"""

    path = output_dir / "model_metadata.txt"
    path.write_text(text.strip())
    print(f"Saved metadata: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="visual_bmi_cropped",
        help="Root of cropped dataset (must contain body/, face/, and the CSV)",
    )
    parser.add_argument(
        "--face_backbone",
        default="outputs_face/face_backbone.weights.h5",
        help="Path to pretrained face backbone weights",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs_v3",
    )
    parser.add_argument("--stage1_epochs", type=int, default=30)
    parser.add_argument("--stage2_epochs", type=int, default=20)

    parser.add_argument("--min_bbox_area_ratio", type=float, default=0.0)
    parser.add_argument("--keep_clusters", type=str, default=None)
    parser.add_argument("--curation_csv", type=str, default=None)

    parser.add_argument(
        "--no_sample_weights",
        action="store_true",
        help="Disable inverse-density BMI sample weighting on the train set.",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=3.0,
        help="Huber loss delta. Larger value penalizes tail errors more quadratically.",
    )
    parser.add_argument(
        "--no_face",
        action="store_true",
        help="Ablation: drop the face branch and train body-only (body image + is_female).",
    )

    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_two_branch_dataframe(args.data_dir)

    keep_clusters = None
    if args.keep_clusters is not None:
        keep_clusters = [int(c.strip()) for c in args.keep_clusters.split(",") if c.strip()]

    df = bmi_model.apply_curation_filters(
        df,
        data_dir=args.data_dir,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        keep_clusters=keep_clusters,
        curation_csv=args.curation_csv,
    )

    train_df, val_df, test_df = bmi_model.split_dataframe(df)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    if args.no_sample_weights:
        train_weights = None
        print("Sample weighting: DISABLED (uniform).")
    else:
        train_weights = compute_bmi_sample_weights(train_df["target_bmi"].values)
        print(
            f"Sample weighting: ENABLED. weight stats — "
            f"min={train_weights.min():.3f}, max={train_weights.max():.3f}, "
            f"mean={train_weights.mean():.3f}"
        )

    use_face = not args.no_face
    if args.no_face:
        print("Architecture: BODY-ONLY (face branch disabled).")
    else:
        print("Architecture: TWO-BRANCH (body + face).")

    train_ds = make_dataset(train_df, training=True, sample_weights=train_weights, use_face=use_face)
    val_ds = make_dataset(val_df, training=False, use_face=use_face)
    test_ds = make_dataset(test_df, training=False, use_face=use_face)

    face_weights = args.face_backbone
    if use_face and face_weights and not Path(face_weights).exists():
        print(f"WARNING: face backbone {face_weights} not found — "
              f"using ImageNet init for face branch instead.")
        face_weights = None
    if not use_face:
        face_weights = None

    model, body_base, face_base = build_two_branch_model(
        face_weights, huber_delta=args.huber_delta, use_face=use_face
    )
    print(f"\nLoss: Huber(delta={args.huber_delta})")
    print("\nModel summary:\n")
    model.summary()

    best_model, ckpt = train(
        model, body_base, face_base,
        train_ds, val_ds, output_dir,
        args.stage1_epochs, args.stage2_epochs,
        huber_delta=args.huber_delta,
    )

    evaluate(best_model, test_ds)
    export_tflite(best_model, output_dir, use_face=use_face)
    save_metadata(output_dir, use_face=use_face)

    print(f"\nDone.\n  Keras: {ckpt}")


if __name__ == "__main__":
    main()
