"""
Extract pose keypoints from each image, derive a person-bbox area ratio, and
cluster the keypoint configurations. The resulting curation.csv lets
bmi_model.apply_curation_filters drop low-quality / bad-posture images before
training.

Inspired by DigitalScale (im-ethz/DigitalScale), which uses person-detection
area-ratio + posture clustering as a data-curation pre-step and reports a
sizeable accuracy gain from it.

Run once per dataset:
    python data_curation.py --data_dir visual_bmi --k 5 --preview

Then during training:
    python bmi_model.py --data_dir visual_bmi \\
        --min_bbox_area_ratio 0.15 \\
        --keep_clusters 2,3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import body_crop


# MediaPipe BlazePose gives 33 landmarks. For clustering we use a stable
# subset (torso + limbs) so cluster centroids reflect posture rather than
# face/foot landmark visibility noise.
CLUSTER_LANDMARKS = [
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
]

VIS_THRESHOLD = 0.3
MIN_VISIBLE_FOR_CLUSTER = 8  # of 12


DETECT_MAX_DIM = 1280


def extract_one(detector, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]

    # MediaPipe Pose can hang on multi-megapixel inputs on macOS Metal. Detect
    # on a downscaled copy; landmarks are normalized so they apply to the
    # original at full resolution.
    scale = DETECT_MAX_DIM / max(h, w)
    if scale < 1.0:
        small = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image
    landmarks = body_crop.detect_landmarks(detector, small)

    record = {"image": image_path.name, "image_h": h, "image_w": w}

    if landmarks is None:
        record["bbox_area_ratio"] = 0.0
        record["mean_visibility"] = 0.0
        record["num_visible"] = 0
        for i in range(33):
            record[f"kp{i:02d}_x"] = np.nan
            record[f"kp{i:02d}_y"] = np.nan
            record[f"kp{i:02d}_v"] = 0.0
        return record

    xs_vis, ys_vis, vs = [], [], []
    for i, lm in enumerate(landmarks):
        x_norm = float(lm.x)
        y_norm = float(lm.y)
        v = float(lm.visibility) if lm.visibility is not None else 1.0
        record[f"kp{i:02d}_x"] = x_norm
        record[f"kp{i:02d}_y"] = y_norm
        record[f"kp{i:02d}_v"] = v
        if v >= VIS_THRESHOLD:
            xs_vis.append(x_norm)
            ys_vis.append(y_norm)
        vs.append(v)

    if xs_vis:
        # Landmark coords are already normalized to image dims by MediaPipe,
        # so the bbox area ratio is just span_x * span_y.
        record["bbox_area_ratio"] = float(
            (max(xs_vis) - min(xs_vis)) * (max(ys_vis) - min(ys_vis))
        )
    else:
        record["bbox_area_ratio"] = 0.0

    record["mean_visibility"] = float(np.mean(vs)) if vs else 0.0
    record["num_visible"] = int(sum(1 for v in vs if v >= VIS_THRESHOLD))
    return record


def extract_all(in_dir, out_csv, overwrite=False, limit=None):
    in_dir = Path(in_dir).resolve()
    images = sorted(
        p for p in in_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in body_crop.IMAGE_EXTENSIONS
    )
    print(f"Found {len(images)} images under {in_dir}")

    cached = {}
    if out_csv.exists() and not overwrite:
        existing = pd.read_csv(out_csv)
        cached = {row["image"]: row.to_dict() for _, row in existing.iterrows()}
        print(f"Loaded {len(cached)} cached entries from {out_csv}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # MediaPipe Pose on macOS Metal accumulates internal state across calls and
    # eventually stalls (a detect call that normally takes 30ms blocks for tens
    # of seconds). Recreate the detector every RECREATE_EVERY calls to dodge it.
    RECREATE_EVERY = 25
    detector = body_crop.make_pose_landmarker()
    calls_since_recreate = 0
    rows = []
    new_count = 0
    try:
        for i, src in enumerate(images):
            if src.name in cached:
                rows.append(cached[src.name])
                continue

            if calls_since_recreate >= RECREATE_EVERY:
                try:
                    detector.close()
                except Exception:
                    pass
                detector = body_crop.make_pose_landmarker()
                calls_since_recreate = 0

            print(f"[{i+1}/{len(images)}] {src.name}", flush=True)
            record = extract_one(detector, src)
            calls_since_recreate += 1
            if record is None:
                continue
            rows.append(record)
            new_count += 1

            if (i + 1) % 250 == 0:
                print(f"  --> {i + 1}/{len(images)} (new this run: {new_count})", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)

            if limit is not None and new_count >= limit:
                print(f"  --> reached limit={limit}, stopping early", flush=True)
                break
    finally:
        try:
            detector.close()
        except Exception:
            pass

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv} ({new_count} newly extracted)")
    return df


def _features_for_clustering(df):
    feats = []
    keep_idx = []
    for idx, row in df.iterrows():
        xs, ys = [], []
        for lm in CLUSTER_LANDMARKS:
            x = row.get(f"kp{lm:02d}_x", np.nan)
            y = row.get(f"kp{lm:02d}_y", np.nan)
            v = row.get(f"kp{lm:02d}_v", 0.0)
            if pd.notna(x) and pd.notna(y) and v >= VIS_THRESHOLD:
                xs.append(x); ys.append(y)
            else:
                xs.append(np.nan); ys.append(np.nan)

        xs = np.array(xs); ys = np.array(ys)
        if np.sum(~np.isnan(xs)) < MIN_VISIBLE_FOR_CLUSTER:
            continue

        x_min, x_max = np.nanmin(xs), np.nanmax(xs)
        y_min, y_max = np.nanmin(ys), np.nanmax(ys)
        if x_max - x_min < 1e-3 or y_max - y_min < 1e-3:
            continue

        # Normalize each pose to its own bbox so the clusters represent
        # posture, not where the person sits in the frame.
        xs_n = (xs - x_min) / (x_max - x_min)
        ys_n = (ys - y_min) / (y_max - y_min)
        xs_n = np.where(np.isnan(xs_n), 0.5, xs_n)
        ys_n = np.where(np.isnan(ys_n), 0.5, ys_n)

        feats.append(np.concatenate([xs_n, ys_n]))
        keep_idx.append(idx)

    return np.array(feats), keep_idx


def cluster_postures(df, k=5, seed=42):
    feats, keep_idx = _features_for_clustering(df)
    print(
        f"Clustering {len(feats)} poses into {k} clusters "
        f"(skipped {len(df) - len(feats)} with too few visible landmarks)"
    )
    if len(feats) < k:
        raise ValueError(f"Only {len(feats)} usable poses but k={k}.")

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(feats)

    df["cluster"] = -1
    for idx, label in zip(keep_idx, labels):
        df.at[idx, "cluster"] = int(label)
    return df


def render_cluster_preview(df, in_dir, out_path, n_per_cluster=10, tile_size=160):
    in_dir = Path(in_dir)
    clusters = sorted(c for c in df["cluster"].unique() if c >= 0)

    canvas = np.full(
        (len(clusters) * tile_size, n_per_cluster * tile_size, 3),
        255,
        dtype=np.uint8,
    )

    for r, c in enumerate(clusters):
        sub = df[df["cluster"] == c].head(n_per_cluster)
        for i, (_, row) in enumerate(sub.iterrows()):
            matches = list(in_dir.rglob(str(row["image"])))
            if not matches:
                continue
            img = cv2.imread(str(matches[0]))
            if img is None:
                continue
            img = cv2.resize(img, (tile_size, tile_size))
            canvas[
                r * tile_size : (r + 1) * tile_size,
                i * tile_size : (i + 1) * tile_size,
            ] = img

        cv2.putText(
            canvas,
            f"cluster {c} (n={int((df['cluster'] == c).sum())})",
            (6, r * tile_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), canvas)
    print(f"Wrote cluster preview: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="visual_bmi")
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Defaults to <data_dir>/curation.csv",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of posture clusters")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a grid of sample images per cluster for manual inspection",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract keypoints even if a row already exists in the cache",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many new images then exit (for chunked runs)",
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Skip clustering and preview rendering",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else data_dir / "curation.csv"

    df = extract_all(data_dir, out_csv, overwrite=args.overwrite, limit=args.limit)
    if args.extract_only:
        print("--extract_only set: skipping clustering and preview.")
        return
    df = cluster_postures(df, k=args.k)
    df.to_csv(out_csv, index=False)
    print(f"Updated {out_csv} with cluster labels.")

    print("\nCluster distribution:")
    print(df["cluster"].value_counts().sort_index())
    print("\nBbox area-ratio quantiles:")
    print(df["bbox_area_ratio"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    if args.preview:
        preview_path = out_csv.parent / "curation_clusters_preview.jpg"
        render_cluster_preview(df, data_dir, preview_path)
        print(
            "\nInspect the preview and pick which clusters look like "
            "'good-quality, full-body' postures, then pass e.g. "
            "--keep_clusters 2,3 when training."
        )


if __name__ == "__main__":
    main()
