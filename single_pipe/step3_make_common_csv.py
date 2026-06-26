"""
Step 3 — Convert both cleaned datasets into one common CSV format.

Output: work/common.csv (schema in sinbmi_lib.COMMON_COLUMNS)
"""

import pandas as pd

import sinbmi_lib as lib


def main():
    work = lib.HERE / "work"
    parts = []
    for name in ("visualbmi_clean.csv", "image2bmi_clean.csv"):
        path = work / name
        if not path.exists():
            raise SystemExit(f"{path} missing — run steps 1 and 2 first.")
        parts.append(pd.read_csv(path))

    df = pd.concat(parts, ignore_index=True)[lib.COMMON_COLUMNS]

    out_path = work / "common.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df)} rows, "
          f"{df['person_id'].nunique()} subjects")
    print(df.groupby("source")["bmi"].describe().round(2))


if __name__ == "__main__":
    main()
