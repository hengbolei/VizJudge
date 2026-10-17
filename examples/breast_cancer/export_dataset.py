"""Export scikit-learn's bundled Breast Cancer Wisconsin (Diagnostic) data to CSV.

The CSV is committed next to this script, so running the example needs no download and
no scikit-learn. Re-run this only to regenerate it:

    uv run --with scikit-learn python examples/breast_cancer/export_dataset.py

Source: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic),
https://doi.org/10.24432/C5DW2B, licensed CC BY 4.0. Shipped with scikit-learn (BSD-3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("breast_cancer.csv"),
    )
    args = parser.parse_args()

    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    frame.columns = [column.replace(" ", "_") for column in frame.columns]
    # sklearn encodes 0 = malignant, 1 = benign; keep readable labels for the report.
    frame["diagnosis"] = frame.pop("target").map(dict(enumerate(bunch.target_names)))
    frame.to_csv(args.output, index=False, float_format="%.6g")
    print(f"Wrote {len(frame)} rows x {frame.shape[1]} columns to {args.output}")


if __name__ == "__main__":
    main()
