"""
Step 5 — Split by subject ID.

70/15/15 train/val/test at the person level over the combined data, so no
person appears in two splits. (The 2DImage2BMI release's own folder split
shares 462 person IDs between train and test and is deliberately ignored.)

Output: work/common_split.csv
"""

import pandas as pd

import sinbmi_lib as lib


def main():
    work = lib.HERE / "work"
    df = pd.read_csv(work / "common_preprocessed.csv")

    df = lib.split_by_subject(df)

    print("Rows / subjects per split and source:")
    summary = df.groupby(["split", "source"]).agg(
        rows=("image_path", "count"),
        subjects=("person_id", "nunique"),
        mean_bmi=("bmi", "mean"),
    ).round(2)
    print(summary)

    # Guard: no subject in more than one split.
    leaks = df.groupby("person_id")["split"].nunique()
    assert (leaks == 1).all(), "subject leakage across splits!"
    print("Leakage check: OK (every subject in exactly one split)")

    out_path = work / "common_split.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
