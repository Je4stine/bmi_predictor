"""
Step 4 — Crop/pad/resize all body images the same way.

The canonical transform (sinbmi_lib.crop_pad_resize) is: crop the fractional
body box -> pad to square with black -> resize to 260x260. Every image in
every dataset — and later your own app images — goes through exactly this.

By default the result is materialized to work/cache_260/ (~15-25 KB per
image, ~150-250 MB total) so training doesn't re-decode multi-megapixel
originals every epoch. The cache is disposable: delete it and re-run this
step to rebuild. Use --no-cache to skip materialization (training then
applies the transform on the fly, slower per epoch, zero extra disk).

Output: work/common_preprocessed.csv (+ work/cache_260/*.jpg unless --no-cache)
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

import sinbmi_lib as lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't write resized copies; record boxes only.")
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    work = lib.HERE / "work"
    df = pd.read_csv(work / "common.csv")

    if args.no_cache:
        df["cache_path"] = ""
        out_path = work / "common_preprocessed.csv"
        df.to_csv(out_path, index=False)
        print(f"No-cache mode: transform will run on the fly. Wrote {out_path}")
        return

    cache_dir = work / "cache_260"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = []
    n_done = n_cached = 0
    for i, row in df.iterrows():
        src = Path(row["image_path"])
        dst = cache_dir / f"{row['source']}_{i:06d}.jpg"
        if not dst.exists():
            with Image.open(src) as im:
                out = lib.crop_pad_resize(
                    im.convert("RGB"),
                    (row["y0"], row["x0"], row["y1"], row["x1"]),
                )
            out.save(dst, quality=args.quality)
            n_done += 1
        else:
            n_cached += 1
        cache_paths.append(str(dst))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)} (new={n_done} cached={n_cached})")

    df["cache_path"] = cache_paths
    out_path = work / "common_preprocessed.csv"
    df.to_csv(out_path, index=False)

    total_mb = sum(p.stat().st_size for p in cache_dir.glob("*.jpg")) / 1e6
    print(f"Wrote {out_path}; cache: {len(cache_paths)} files, {total_mb:.0f} MB "
          f"(delete work/cache_260 to reclaim — rebuildable)")


if __name__ == "__main__":
    main()
