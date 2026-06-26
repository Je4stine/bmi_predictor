"""
Cache body- and face-cropped versions of the Visual BMI dataset using one pass
of MediaPipe Pose (the face crops come from face-region pose landmarks 0..10).

Output layout:
    <out_dir>/body/<rel_path>.jpg   - body crop (original is copied through if
                                       pose detection failed)
    <out_dir>/face/<rel_path>.jpg   - face crop (only written when a usable
                                       face landmark cluster was found)
    <out_dir>/...csv                 - any CSVs in the input are mirrored

Usage:
    python body_crop.py --in_dir visual_bmi --out_dir visual_bmi_cropped
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_CACHE = Path.home() / ".cache" / "mediapipe" / "pose_landmarker_lite.task"


def ensure_pose_model():
    if POSE_MODEL_CACHE.exists():
        return POSE_MODEL_CACHE
    POSE_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {POSE_MODEL_CACHE} ...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_CACHE)
    return POSE_MODEL_CACHE


def make_pose_landmarker():
    model_path = ensure_pose_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# Face-region pose landmarks: nose (0), eyes (1-6), ears (7-8), mouth (9-10).
FACE_LANDMARK_INDICES = list(range(11))


def detect_landmarks(detector, image_bgr):
    """Returns the list of pose landmarks for the first detected person, or None."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return None

    return result.pose_landmarks[0]


def _visible_xy(landmarks, indices, image_shape, vis_threshold):
    h, w = image_shape[:2]
    xs, ys = [], []

    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        if lm.visibility is not None and lm.visibility < vis_threshold:
            continue
        xs.append(lm.x * w)
        ys.append(lm.y * h)

    return xs, ys


def crop_to_body(landmarks, image_bgr, pad_frac=0.15, min_visible=6, vis_threshold=0.3):
    """
    Returns a body-cropped copy of `image_bgr` from `landmarks`, or None if no
    usable subset of landmarks was visible.
    """
    h, w = image_bgr.shape[:2]
    indices = list(range(len(landmarks)))
    xs, ys = _visible_xy(landmarks, indices, image_bgr.shape, vis_threshold)

    if len(xs) < min_visible:
        return None

    x_min = max(0, int(min(xs) - pad_frac * w))
    x_max = min(w, int(max(xs) + pad_frac * w))
    y_min = max(0, int(min(ys) - pad_frac * h))
    y_max = min(h, int(max(ys) + pad_frac * h))

    if x_max - x_min < 64 or y_max - y_min < 64:
        return None

    return image_bgr[y_min:y_max, x_min:x_max]


def crop_to_face(landmarks, image_bgr, pad_frac=0.4, min_visible=5, vis_threshold=0.3):
    """
    Returns a square face crop derived from face-region pose landmarks, or
    None if too few face landmarks are visible.
    """
    h, w = image_bgr.shape[:2]
    xs, ys = _visible_xy(
        landmarks, FACE_LANDMARK_INDICES, image_bgr.shape, vis_threshold
    )

    if len(xs) < min_visible:
        return None

    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    half = span * (0.5 + pad_frac)

    x_min = max(0, int(cx - half))
    x_max = min(w, int(cx + half))
    y_min = max(0, int(cy - half))
    y_max = min(h, int(cy + half))

    if x_max - x_min < 48 or y_max - y_min < 48:
        return None

    return image_bgr[y_min:y_max, x_min:x_max]


def mirror_csvs(in_dir, out_dirs):
    """
    Copy any CSVs from in_dir into each output directory at the same relative
    path. Used so that each subtree (body/, face/, and the root) is a
    self-contained dataset folder.
    """
    csvs = list(in_dir.rglob("*.csv"))
    for csv_path in csvs:
        rel = csv_path.relative_to(in_dir)
        for out_dir in out_dirs:
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, target)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_dir",
        type=str,
        default="visual_bmi",
        help="Source dataset folder",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="visual_bmi_cropped",
        help="Where to write cropped images",
    )

    parser.add_argument(
        "--body_pad",
        type=float,
        default=0.15,
        help="Padding around the body, as a fraction of image dimensions",
    )

    parser.add_argument(
        "--face_pad",
        type=float,
        default=0.4,
        help="Padding around the face landmarks, as a fraction of face span",
    )

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=92,
        help="JPEG quality for written images",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute crops even if the output file already exists",
    )

    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {in_dir}")

    body_dir = out_dir / "body"
    face_dir = out_dir / "face"
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    mirror_csvs(in_dir, [out_dir, body_dir, face_dir])

    images = sorted(
        p for p in in_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Found {len(images)} images under {in_dir}")
    print(f"Writing body crops to {body_dir}")
    print(f"Writing face crops to {face_dir}")

    pose = make_pose_landmarker()

    n_body_cropped = 0
    n_body_fallback = 0
    n_face_cropped = 0
    n_face_missing = 0
    n_skipped = 0
    n_cached = 0

    try:
        for i, src in enumerate(images):
            rel = src.relative_to(in_dir)
            body_dst = body_dir / rel
            face_dst = face_dir / rel

            body_done = body_dst.exists() and not args.overwrite
            face_done = face_dst.exists() and not args.overwrite

            if body_done and face_done:
                n_cached += 1
                continue

            body_dst.parent.mkdir(parents=True, exist_ok=True)
            face_dst.parent.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(src))

            if image is None:
                n_skipped += 1
                continue

            landmarks = detect_landmarks(pose, image)

            if not body_done:
                body = (
                    crop_to_body(landmarks, image, pad_frac=args.body_pad)
                    if landmarks is not None
                    else None
                )
                if body is None:
                    body = image
                    n_body_fallback += 1
                else:
                    n_body_cropped += 1

                cv2.imwrite(
                    str(body_dst),
                    body,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                )

            if not face_done:
                face = (
                    crop_to_face(landmarks, image, pad_frac=args.face_pad)
                    if landmarks is not None
                    else None
                )
                if face is None:
                    n_face_missing += 1
                else:
                    n_face_cropped += 1
                    cv2.imwrite(
                        str(face_dst),
                        face,
                        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                    )

            if (i + 1) % 250 == 0:
                print(
                    f"  {i + 1}/{len(images)}  "
                    f"body[ok={n_body_cropped} fallback={n_body_fallback}] "
                    f"face[ok={n_face_cropped} miss={n_face_missing}] "
                    f"skip={n_skipped} cached={n_cached}"
                )
    finally:
        try:
            pose.close()
        except Exception:
            pass

    print()
    print(
        f"Done. body[cropped={n_body_cropped} fallback={n_body_fallback}] "
        f"face[cropped={n_face_cropped} missing={n_face_missing}] "
        f"skipped={n_skipped} cached={n_cached}"
    )
    print(f"Body crops: {body_dir}")
    print(f"Face crops: {face_dir}")


if __name__ == "__main__":
    main()
