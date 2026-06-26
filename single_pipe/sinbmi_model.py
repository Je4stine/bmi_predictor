"""
SinBMI model and TF data pipeline, shared by steps 6, 7 and 10.

Architecture (SinBMI, Technion SIPL 2025): EfficientNet-B2 backbone followed
by a seven-layer MLP, each layer a linear transform + GELU, reducing the
feature vector to a scalar BMI. Trained end-to-end with MSE loss, Adam, and
plateau LR scheduling. Body image only — no face branch, no gender input.

Input contract: [1, 260, 260, 3] float32 RGB in 0..1 -> [1, 1] float32 BMI.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import sinbmi_lib as lib

IMG_SIZE = lib.IMG_SIZE
BATCH_SIZE = 16
SEED = lib.SEED

MLP_WIDTHS = [512, 256, 128, 64, 32, 16]  # + final Dense(1) = 7 FC layers


def build_sinbmi(weights="imagenet"):
    base = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=weights,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="body")
    x = inp * 255.0  # Keras EfficientNet normalizes internally from 0..255
    # training=False pins BatchNorm to inference statistics even when the
    # backbone is unfrozen; otherwise unfreezing flips BN to batch statistics
    # and the trained head's inputs shift wholesale (train MAE 2.6 -> 22).
    x = base(x, training=False)
    for width in MLP_WIDTHS:
        x = tf.keras.layers.Dense(width, activation="gelu")(x)
    out = tf.keras.layers.Dense(1, name="bmi")(x)

    model = tf.keras.Model(inp, out, name="sinbmi_effnetb2")
    return model, base


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _decode_cached(path, bmi):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return tf.cast(image, tf.float32) / 255.0, tf.cast(bmi, tf.float32)


def _decode_original(path, box, bmi):
    """On-the-fly variant of the step-4 transform (no cache)."""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])

    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    y0 = tf.clip_by_value(tf.cast(box[0] * shape[0], tf.int32), 0, tf.shape(image)[0] - 1)
    x0 = tf.clip_by_value(tf.cast(box[1] * shape[1], tf.int32), 0, tf.shape(image)[1] - 1)
    h = tf.clip_by_value(tf.cast((box[2] - box[0]) * shape[0], tf.int32),
                         1, tf.shape(image)[0] - y0)
    w = tf.clip_by_value(tf.cast((box[3] - box[1]) * shape[1], tf.int32),
                         1, tf.shape(image)[1] - x0)
    image = tf.image.crop_to_bounding_box(image, y0, x0, h, w)
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)  # pad black
    return tf.cast(image, tf.float32) / 255.0, tf.cast(bmi, tf.float32)


def _augment(image, bmi):
    """SinBMI augmentations: horizontal flip, Gaussian noise, rotation <=10°."""
    image = tf.image.random_flip_left_right(image)

    angle = tf.random.uniform([], -10.0, 10.0) * np.pi / 180.0
    image = _rotate(image, angle)

    noise = tf.random.normal(tf.shape(image), stddev=0.02)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image, bmi


def _rotate(image, angle):
    """Small-angle rotation via projective transform (no tfa dependency)."""
    cos, sin = tf.cos(angle), tf.sin(angle)
    cx = cy = IMG_SIZE / 2.0
    transform = tf.stack([
        cos, -sin, cx - cx * cos + cy * sin,
        sin, cos, cy - cx * sin - cy * cos,
        0.0, 0.0,
    ])
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=transform[None, ...],
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], tf.int32),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )[0]


def make_dataset(df, training=True):
    use_cache = "cache_path" in df.columns and bool(
        df["cache_path"].astype(str).str.len().gt(0).all()
    )
    labels = df["bmi"].astype("float32").values

    if use_cache:
        ds = tf.data.Dataset.from_tensor_slices(
            (df["cache_path"].astype(str).values, labels)
        )
        decode = _decode_cached
    else:
        boxes = df[["y0", "x0", "y1", "x1"]].values.astype("float32")
        ds = tf.data.Dataset.from_tensor_slices(
            (df["image_path"].astype(str).values, boxes, labels)
        )
        decode = _decode_original

    if training:
        # Shuffle paths before decoding: a post-decode buffer holds float32
        # images (~0.8 GB at 1000) and forces 8 GB machines into swap.
        ds = ds.shuffle(len(df), seed=SEED)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Training / export
# ---------------------------------------------------------------------------

def fit(model, train_ds, val_ds, output_dir, epochs=40, patience=10,
        history_name="history.csv", best_threshold=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "sinbmi_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_mae", save_best_only=True,
            mode="min", verbose=1,
            initial_value_threshold=best_threshold,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.3, patience=4,
            min_lr=1e-7, mode="min", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=patience,
            restore_best_weights=True, mode="min", verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / history_name)),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=callbacks)
    # Load best weights into the live model rather than load_model(ckpt):
    # a second resident EfficientNet pushes an 8 GB machine into swap.
    model.load_weights(ckpt)
    return model, ckpt


def fit_two_phase(model, base, make_train_ds, make_val_ds, output_dir,
                  head_epochs=15, epochs=40, head_lr=1e-3, ft_lr=1e-5,
                  patience=10):
    """
    Phase 1 trains only the MLP head with the backbone frozen; a randomly
    initialized head sends large gradients through the backbone otherwise.
    Phase 2 then unfreezes and fine-tunes end-to-end at a low learning rate
    (BatchNorm stays in inference mode throughout — see build_sinbmi). The
    shared checkpoint only ever improves: phase 2 saves nothing unless it
    beats phase 1's best val_mae.

    make_train_ds / make_val_ds are zero-arg callables; each phase gets
    fresh dataset objects (reusing one across fit() calls triggers spurious
    "input ran out of data" truncation).
    """
    output_dir = Path(output_dir)

    print(f"Phase 1/2: frozen backbone, head only "
          f"({head_epochs} epochs, lr={head_lr:g})")
    base.trainable = False
    compile_model(model, lr=head_lr)
    best, ckpt = fit(model, make_train_ds(), make_val_ds(), output_dir,
                     epochs=head_epochs, patience=patience,
                     history_name="history_phase1.csv")
    phase1_best = float(
        pd.read_csv(output_dir / "history_phase1.csv")["val_mae"].min()
    )

    print(f"Phase 2/2: end-to-end fine-tune ({epochs} epochs, lr={ft_lr:g}), "
          f"val_mae to beat: {phase1_best:.3f}")
    base.trainable = True
    compile_model(best, lr=ft_lr)
    return fit(best, make_train_ds(), make_val_ds(), output_dir,
               epochs=epochs, patience=patience,
               history_name="history_phase2.csv",
               best_threshold=phase1_best)


def export_tflite(model, output_dir):
    output_dir = Path(output_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    path = output_dir / "sinbmi_body.tflite"
    path.write_bytes(converter.convert())

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    detail = interpreter.get_input_details()[0]
    print(f"TFLite saved: {path}")
    print(f"  input  {detail['name']} shape={list(detail['shape'])} float32 RGB 0..1")
    print(f"  output [1, 1] float32 BMI")
    return path


def load_split(split):
    df = pd.read_csv(lib.HERE / "work" / "common_split.csv")
    return df[df["split"] == split].reset_index(drop=True)


def resolve_checkpoint(path):
    """Accept either a .keras file or a run directory containing one."""
    path = Path(path)
    if path.is_dir():
        path = path / "sinbmi_best.keras"
    return path if path.exists() else None
