import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42


def find_column(df, possible_names):
    """
    Finds a column in the CSV using a list of possible names.
    """
    normalized_columns = {
        col.lower().strip(): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower().strip()
        if key in normalized_columns:
            return normalized_columns[key]

    return None


def find_image_path(root_dir, filename):
    """
    Attempts to find an image file inside the dataset folder.
    Handles filenames, relative paths, absolute-like paths, and Kaggle-style paths.
    """
    root_dir = Path(root_dir)
    raw_value = str(filename).strip()

    raw_value = raw_value.replace("\\", "/")

    kaggle_prefixes = [
        "/kaggle/input/visual-bmi/",
        "kaggle/input/visual-bmi/",
        "/kaggle/input/",
        "kaggle/input/",
    ]

    for prefix in kaggle_prefixes:
        if raw_value.startswith(prefix):
            raw_value = raw_value[len(prefix):]

    relative_value = raw_value.lstrip("/")

    possible_paths = [
        root_dir / relative_value,
        root_dir / "bodyface_1to17" / relative_value,
        root_dir / Path(relative_value).name,
        root_dir / "bodyface_1to17" / Path(relative_value).name,
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    basename = Path(relative_value).name

    if basename:
        matches = list(root_dir.rglob(basename))
        if matches:
            return str(matches[0])

    return None


def is_valid_image(image_path):
    """
    Returns True if the image can be opened and decoded.
    Removes corrupt/truncated images before TensorFlow sees them.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            img.convert("RGB")

        return True

    except Exception as e:
        print(f"Skipping corrupted image: {image_path} | Error: {e}")
        return False


def load_and_prepare_dataframe(data_dir):
    data_dir = Path(data_dir)

    csv_candidates = [
        data_dir / "visual_bmi_annotations.csv",
        data_dir / "VisualBMI.csv",
        data_dir / "annotations.csv",
        data_dir / "visual-body-to-bmi.csv",
    ]

    csv_path = None

    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        csv_files = list(data_dir.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found inside {data_dir}. "
                f"Expected visual_bmi_annotations.csv"
            )

        csv_path = csv_files[0]

    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    print("CSV columns:")
    print(list(df.columns))
    print(f"Rows before cleaning: {len(df)}")

    image_col = find_column(
        df,
        [
            "image_path",
            "image",
            "filename",
            "file",
            "file_name",
            "name",
            "path",
            "img",
        ],
    )

    bmi_col = find_column(
        df,
        [
            "BMI",
            "bmi",
            "body_mass_index",
            "Body Mass Index",
        ],
    )

    height_col = find_column(
        df,
        [
            "height_in",
            "height",
            "height_cm",
            "Height",
            "Height(cm)",
            "height_in_cm",
            "stature",
        ],
    )

    weight_col = find_column(
        df,
        [
            "weight_lb",
            "weight",
            "weight_kg",
            "Weight",
            "Weight(kg)",
            "body_weight",
        ],
    )

    if image_col is None:
        raise ValueError(
            "Could not find image filename/path column. "
            "Please check the CSV columns printed above."
        )

    if bmi_col is None:
        if height_col is None or weight_col is None:
            raise ValueError(
                "Could not find BMI column, and could not find both height and weight columns."
            )

        print(
            f"Computing BMI using height column '{height_col}' "
            f"and weight column '{weight_col}'"
        )

        height = pd.to_numeric(df[height_col], errors="coerce")
        weight = pd.to_numeric(df[weight_col], errors="coerce")

        # Unit guessing
        if "lb" in weight_col.lower():
            weight_kg = weight * 0.45359237
        else:
            weight_kg = weight

        if "in" in height_col.lower():
            height_m = height * 0.0254
        elif height.dropna().mean() > 3:
            height_m = height / 100.0
        else:
            height_m = height

        df["target_bmi"] = weight_kg / (height_m ** 2)

    else:
        print(f"Using BMI column: {bmi_col}")
        df["target_bmi"] = pd.to_numeric(df[bmi_col], errors="coerce")

    print(f"Using image column: {image_col}")

    is_female_col = find_column(
        df,
        ["is_female", "isFemale", "female", "gender_female"],
    )

    if is_female_col is not None:
        print(f"Using auxiliary gender column: {is_female_col}")

        def _parse_is_female(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, (bool, np.bool_)):
                return 1.0 if bool(value) else 0.0
            text = str(value).strip().lower()
            if text in ("true", "1", "1.0", "yes", "y", "f", "female"):
                return 1.0
            if text in ("false", "0", "0.0", "no", "n", "m", "male"):
                return 0.0
            return np.nan

        df["aux_is_female"] = df[is_female_col].apply(_parse_is_female).astype("float32")
    else:
        print("No is_female column found — auxiliary gender input will be disabled.")
        df["aux_is_female"] = np.nan

    image_paths = []

    for value in df[image_col]:
        image_path = find_image_path(data_dir, value)
        image_paths.append(image_path)

    df["image_path_resolved"] = image_paths

    df = df.dropna(subset=["image_path_resolved", "target_bmi"])

    df = df[
        df["image_path_resolved"].apply(
            lambda p: p is not None and os.path.exists(p)
        )
    ]

    df = df[(df["target_bmi"] >= 10) & (df["target_bmi"] <= 60)]

    print(f"Rows before image validation: {len(df)}")

    if os.environ.get("BMI_SKIP_IMAGE_VALIDATION") == "1":
        print("Skipping PIL image validation (BMI_SKIP_IMAGE_VALIDATION=1)")
    else:
        df = df[df["image_path_resolved"].apply(is_valid_image)]

    print(f"Rows after image validation: {len(df)}")
    print(f"Rows after cleaning: {len(df)}")

    if len(df) < 100:
        raise ValueError(
            "Very few usable rows found. Check that image paths match your image folder."
        )

    df = df.reset_index(drop=True)

    return df


def _find_curation_csv(data_dir):
    """
    Look for curation.csv in data_dir or its immediate parent. The two-branch
    pipeline trains on visual_bmi_cropped/body, but the curation was computed
    on visual_bmi/, so we also check the grandparent.
    """
    data_dir = Path(data_dir)
    for candidate in (
        data_dir / "curation.csv",
        data_dir.parent / "curation.csv",
        data_dir.parent.parent / "curation.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def apply_curation_filters(
    df,
    data_dir,
    min_bbox_area_ratio=0.0,
    keep_clusters=None,
    curation_csv=None,
):
    """
    Merge curation.csv (produced by data_curation.py) by image basename and
    apply quality/posture filters. Adds 'bbox_area_ratio' and
    'posture_cluster' columns to the returned df.
    """
    if curation_csv is None:
        curation_csv = _find_curation_csv(data_dir)
    else:
        curation_csv = Path(curation_csv)

    if curation_csv is None or not curation_csv.exists():
        if min_bbox_area_ratio > 0 or keep_clusters is not None:
            print(
                "WARNING: curation filters requested but no curation.csv found. "
                "Run data_curation.py first, or pass --curation_csv."
            )
        return df

    print(f"Loading curation data from: {curation_csv}")
    cur = pd.read_csv(curation_csv)
    cur_by_name = {row["image"]: row for _, row in cur.iterrows()}

    def _get(name, field, default):
        row = cur_by_name.get(name)
        if row is None:
            return default
        value = row.get(field, default)
        return default if pd.isna(value) else value

    basenames = df["image_path_resolved"].apply(lambda p: Path(p).name)
    df["bbox_area_ratio"] = basenames.apply(
        lambda n: float(_get(n, "bbox_area_ratio", np.nan))
    )
    df["posture_cluster"] = basenames.apply(
        lambda n: int(_get(n, "cluster", -1))
    )

    matched = df["bbox_area_ratio"].notna().sum()
    print(f"Matched curation rows: {matched}/{len(df)}")

    before = len(df)
    if min_bbox_area_ratio > 0.0:
        df = df[df["bbox_area_ratio"] >= min_bbox_area_ratio]
        print(f"After bbox_area_ratio >= {min_bbox_area_ratio}: {len(df)}")

    if keep_clusters is not None:
        keep_clusters = list(keep_clusters)
        df = df[df["posture_cluster"].isin(keep_clusters)]
        print(f"After keep_clusters {keep_clusters}: {len(df)}")

    print(f"Curation filters: {before} -> {len(df)} rows")
    return df.reset_index(drop=True)


def split_dataframe(df):
    """
    Splits into train, validation, and test.
    Uses person_id/subject_id where available to avoid data leakage.
    """
    subject_col = find_column(
        df,
        [
            "person_id",
            "subject_id",
            "id",
            "ID",
            "user_id",
            "individual_id",
        ],
    )

    if subject_col is not None:
        print(f"Splitting by subject column: {subject_col}")

        subjects = df[subject_col].dropna().unique()

        train_subjects, temp_subjects = train_test_split(
            subjects,
            test_size=0.30,
            random_state=SEED,
        )

        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            test_size=0.50,
            random_state=SEED,
        )

        train_df = df[df[subject_col].isin(train_subjects)]
        val_df = df[df[subject_col].isin(val_subjects)]
        test_df = df[df[subject_col].isin(test_subjects)]

    else:
        print("No subject/person ID column found. Using image-level split.")

        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=SEED,
            shuffle=True,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=SEED,
            shuffle=True,
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")

    return train_df, val_df, test_df


def _decode_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_image(path, bmi):
    return _decode_image(path), tf.cast(bmi, tf.float32)


def load_image_aux(path, aux, bmi):
    image = _decode_image(path)
    aux = tf.reshape(tf.cast(aux, tf.float32), [1])
    bmi = tf.cast(bmi, tf.float32)
    return (image, aux), bmi


def random_erase(image, p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)):
    """
    Cutout / random erasing: zeros out a random rectangle of the image with
    probability `p`. Helps the model not over-rely on any single body region.
    """
    if tf.random.uniform([]) > p:
        return image

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    area = tf.cast(h * w, tf.float32)

    target_area = area * tf.random.uniform([], scale[0], scale[1])
    aspect = tf.random.uniform([], ratio[0], ratio[1])

    erase_h = tf.cast(tf.round(tf.sqrt(target_area * aspect)), tf.int32)
    erase_w = tf.cast(tf.round(tf.sqrt(target_area / aspect)), tf.int32)

    erase_h = tf.minimum(erase_h, h - 1)
    erase_w = tf.minimum(erase_w, w - 1)

    y = tf.random.uniform([], 0, h - erase_h, dtype=tf.int32)
    x = tf.random.uniform([], 0, w - erase_w, dtype=tf.int32)

    mask = tf.pad(
        tf.zeros((erase_h, erase_w, 1), dtype=image.dtype),
        [[y, h - y - erase_h], [x, w - x - erase_w], [0, 0]],
        constant_values=1,
    )

    return image * mask


_AUGMENT_LAYERS = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=0.04, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.05),
            width_factor=(-0.10, 0.05),
            fill_mode="reflect",
        ),
    ],
    name="geom_augment",
)


def _augment_image_only(image):
    image = tf.image.random_flip_left_right(image)
    image = _AUGMENT_LAYERS(image, training=True)

    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.image.random_hue(image, max_delta=0.03)

    image = tf.clip_by_value(image, 0.0, 1.0)
    image = random_erase(image, p=0.25)
    return image


def augment_image(image, bmi):
    return _augment_image_only(image), bmi


def augment_image_aux(inputs, bmi):
    image, aux = inputs
    return (_augment_image_only(image), aux), bmi


def make_dataset(dataframe, training=True, with_aux=False):
    paths = dataframe["image_path_resolved"].astype(str).values
    labels = dataframe["target_bmi"].astype("float32").values

    if with_aux:
        aux = dataframe["aux_is_female"].astype("float32").values
        dataset = tf.data.Dataset.from_tensor_slices((paths, aux, labels))
        dataset = dataset.map(load_image_aux, num_parallel_calls=tf.data.AUTOTUNE)
        aug_fn = augment_image_aux
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        aug_fn = augment_image

    if training:
        dataset = dataset.shuffle(1000, seed=SEED)
        dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _se_block(x, ratio=16, name="se"):
    """Channel-wise Squeeze-and-Excitation gate applied to a 4-D feature map."""
    channels = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_squeeze")(x)
    se = tf.keras.layers.Dense(
        max(channels // ratio, 8), activation="relu", name=f"{name}_reduce"
    )(se)
    se = tf.keras.layers.Dense(channels, activation="sigmoid", name=f"{name}_excite")(se)
    se = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return tf.keras.layers.Multiply(name=f"{name}_scale")([x, se])


BACKBONES = {
    "efficientnetv2b0": {
        "ctor": lambda: tf.keras.applications.EfficientNetV2B0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
            include_preprocessing=True,
        ),
        # EfficientNetV2 expects 0..255; rescaling is built in.
        "preprocess": lambda x: x * 255.0,
        "finetune_layers": 60,
        "use_se": False,
    },
    "mobilenetv2": {
        "ctor": lambda: tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        ),
        "preprocess": lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(
            x * 255.0
        ),
        "finetune_layers": 30,
        "use_se": False,
    },
    "densenet121": {
        "ctor": lambda: tf.keras.applications.DenseNet121(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        ),
        "preprocess": lambda x: tf.keras.applications.densenet.preprocess_input(
            x * 255.0
        ),
        "finetune_layers": 60,
        "use_se": False,
    },
    "densenet121_se": {
        "ctor": lambda: tf.keras.applications.DenseNet121(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        ),
        "preprocess": lambda x: tf.keras.applications.densenet.preprocess_input(
            x * 255.0
        ),
        "finetune_layers": 60,
        "use_se": True,
    },
}


def build_model(backbone_name="efficientnetv2b0", use_aux=False):
    if backbone_name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Choose from {list(BACKBONES)}"
        )

    spec = BACKBONES[backbone_name]
    base_model = spec["ctor"]()
    base_model.trainable = False

    image_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = spec["preprocess"](image_input)
    x = base_model(x, training=False)

    if spec.get("use_se", False):
        x = _se_block(x, name="head_se")

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if use_aux:
        aux_input = tf.keras.Input(shape=(1,), name="is_female")
        x = tf.keras.layers.Concatenate()([x, aux_input])
        model_inputs = [image_input, aux_input]
    else:
        model_inputs = image_input

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="linear", name="bmi")(x)

    model = tf.keras.Model(model_inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ],
    )

    model._backbone_name = backbone_name
    model._use_aux = use_aux
    return model, base_model


def train_model(
    model,
    base_model,
    train_ds,
    val_ds,
    output_dir,
    stage1_epochs,
    stage2_epochs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_bmi_model.keras"

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
            patience=8,
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

    print("\nStage 1: Training regression head...\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks,
    )

    backbone_name = getattr(model, "_backbone_name", "efficientnetv2b0")
    n_unfreeze = BACKBONES[backbone_name]["finetune_layers"]

    print(f"\nStage 2: Fine-tuning last {n_unfreeze} layers of {backbone_name}...\n")

    base_model.trainable = True

    for layer in base_model.layers[:-n_unfreeze]:
        layer.trainable = False

    # Keep BatchNorm in inference mode even when unfrozen — updating BN running
    # stats from small fine-tune batches reliably hurts on transfer learning.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.Huber(delta=1.0),
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


def evaluate_model(model, test_ds):
    print("\nEvaluating on test set...\n")

    results = model.evaluate(test_ds, return_dict=True)

    print("\nTest results:")

    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("\nSample predictions:")

    for inputs, labels in test_ds.take(1):
        predictions = model.predict(inputs, verbose=0).flatten()

        for actual, predicted in zip(labels.numpy()[:10], predictions[:10]):
            print(f"Actual BMI: {actual:.2f} | Predicted BMI: {predicted:.2f}")


def convert_to_tflite(model, output_dir):
    output_dir = Path(output_dir)

    tflite_path = output_dir / "bmi_model.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"\nSaved TensorFlow Lite model to: {tflite_path}")

    return tflite_path


def test_tflite_model(tflite_path, test_df):
    print("\nTesting TensorFlow Lite model...\n")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite input details:")
    print(input_details)

    print("TFLite output details:")
    print(output_details)

    sample_row = test_df.sample(1, random_state=SEED).iloc[0]

    image_path = sample_row["image_path_resolved"]
    actual_bmi = sample_row["target_bmi"]

    image_raw = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image_raw,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0).numpy().astype(np.float32)

    aux_value = sample_row.get("aux_is_female")

    for detail in input_details:
        # Identify each input tensor by its shape: image is 4-D, aux is 2-D.
        rank = len(detail["shape"])

        if rank == 4:
            interpreter.set_tensor(detail["index"], image)
        else:
            if aux_value is None or pd.isna(aux_value):
                aux_value = 0.0
            aux_array = np.array([[float(aux_value)]], dtype=np.float32)
            interpreter.set_tensor(detail["index"], aux_array)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    predicted_bmi = float(output[0][0])

    print(f"Sample image: {image_path}")
    print(f"Actual BMI: {actual_bmi:.2f}")
    if len(input_details) > 1:
        print(f"is_female: {aux_value}")
    print(f"TFLite predicted BMI: {predicted_bmi:.2f}")


def save_model_metadata(output_dir, use_aux=False, backbone="efficientnetv2b0"):
    output_dir = Path(output_dir)

    aux_section = (
        f"""
Aux input (gender):
- Name: is_female
- Shape: [1, 1]
- Type: float32
- Encoding: 1.0 = female, 0.0 = male
- Note: identify the input tensor by rank — the image input is 4-D, this is 2-D.
"""
        if use_aux
        else "\nAux input: none. Model takes a single image input.\n"
    )

    metadata = f"""
BMI model metadata

Task:
- Image-to-BMI regression

Backbone:
- {backbone}

Dataset:
- Kaggle Visual BMI dataset

Image input:
- Name: image
- Shape: [1, {IMG_SIZE}, {IMG_SIZE}, 3]
- Type: float32
- Range: 0.0 to 1.0
- Color format: RGB
{aux_section}
Output:
- Shape: [1, 1]
- Type: float32
- Meaning: predicted BMI value

Android preprocessing:
- Resize bitmap to {IMG_SIZE}x{IMG_SIZE}
- Convert RGB pixels to floats
- Divide each R/G/B value by 255.0
{"- Pass is_female as float32 1.0 (female) or 0.0 (male)." if use_aux else ""}

Important:
- This model gives an estimated BMI from image.
- It is not a medical diagnosis.
- Accuracy may be affected by camera angle, pose, clothing, lighting, and body type.
"""

    path = output_dir / "model_metadata.txt"
    path.write_text(metadata.strip())

    print(f"Saved metadata to: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="visual_bmi",
        help="Path to Kaggle Visual BMI dataset folder",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Folder where trained model and TFLite model will be saved",
    )

    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=40,
        help="Epochs for training the regression head",
    )

    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=20,
        help="Epochs for fine-tuning the backbone",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnetv2b0",
        choices=list(BACKBONES),
        help="Image backbone to use",
    )

    parser.add_argument(
        "--use_aux",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help=(
            "Whether to use is_female as an auxiliary input. "
            "'auto' enables it when the column is present in the CSV."
        ),
    )

    parser.add_argument(
        "--min_bbox_area_ratio",
        type=float,
        default=0.0,
        help=(
            "Drop training rows whose person-bbox area ratio (from "
            "curation.csv) is below this threshold. 0.0 disables the filter."
        ),
    )

    parser.add_argument(
        "--keep_clusters",
        type=str,
        default=None,
        help=(
            "Comma-separated list of posture clusters to keep (e.g. '2,3'). "
            "Run data_curation.py --preview to choose."
        ),
    )

    parser.add_argument(
        "--curation_csv",
        type=str,
        default=None,
        help="Explicit path to curation.csv. Auto-detected if omitted.",
    )

    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_dataframe(args.data_dir)

    keep_clusters = None
    if args.keep_clusters is not None:
        keep_clusters = [int(c.strip()) for c in args.keep_clusters.split(",") if c.strip()]

    df = apply_curation_filters(
        df,
        data_dir=args.data_dir,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        keep_clusters=keep_clusters,
        curation_csv=args.curation_csv,
    )

    has_aux_column = df["aux_is_female"].notna().any()

    if args.use_aux == "yes":
        use_aux = True
    elif args.use_aux == "no":
        use_aux = False
    else:
        use_aux = has_aux_column

    if use_aux:
        if not has_aux_column:
            raise ValueError(
                "--use_aux=yes was set but no usable is_female column was found."
            )
        before = len(df)
        df = df.dropna(subset=["aux_is_female"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with missing is_female.")
        print(f"Auxiliary gender input ENABLED. Rows: {len(df)}")
    else:
        print("Auxiliary gender input DISABLED.")

    train_df, val_df, test_df = split_dataframe(df)

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    train_ds = make_dataset(train_df, training=True, with_aux=use_aux)
    val_ds = make_dataset(val_df, training=False, with_aux=use_aux)
    test_ds = make_dataset(test_df, training=False, with_aux=use_aux)

    model, base_model = build_model(backbone_name=args.backbone, use_aux=use_aux)

    print("\nModel summary:\n")
    model.summary()

    best_model, checkpoint_path = train_model(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=output_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )

    print(f"\nBest Keras model saved at: {checkpoint_path}")

    evaluate_model(best_model, test_ds)

    tflite_path = convert_to_tflite(best_model, output_dir)

    test_tflite_model(tflite_path, test_df)

    save_model_metadata(output_dir, use_aux=use_aux, backbone=args.backbone)

    print("\nDone.")
    print(f"Keras model: {checkpoint_path}")
    print(f"TFLite model: {tflite_path}")


if __name__ == "__main__":
    main()