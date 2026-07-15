"""Read all PSV patient files from training_setA + training_setB into one combined
parquet cache, tagged with patient_id and hospital.

Usage:
    python3 01_build_dataset.py                 # full 40,336-patient run
    python3 01_build_dataset.py --n-patients 200 # small slice for a fast end-to-end test
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import config


def _read_one_file(args) -> pd.DataFrame:
    path, hospital = args
    df = pd.read_csv(path, sep="|")  # pandas' default na_values already includes the literal "NaN"
    df["patient_id"] = int(Path(path).stem[1:])  # e.g. "p000902" -> 902; unique across A and B
    df["hospital"] = hospital
    return df


def _list_files(data_dir: Path, n_patients: int | None):
    files = sorted(data_dir.glob("*.psv"))
    if n_patients is not None:
        files = files[:n_patients]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=None,
                        help="Only read the first N files per hospital (for a fast smoke test).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files_a = _list_files(config.DATA_DIR_A, args.n_patients)
    files_b = _list_files(config.DATA_DIR_B, args.n_patients)
    print(f"Found {len(files_a)} hospital-A files, {len(files_b)} hospital-B files.")

    tasks = [(p, "A") for p in files_a] + [(p, "B") for p in files_b]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        dfs = list(executor.map(_read_one_file, tasks))

    combined = pd.concat(dfs, ignore_index=True)

    # Sanity: confirm the literal "NaN" string in the source files was parsed as missing.
    assert combined["HR"].dtype.kind == "f", "HR column did not parse as float/NaN as expected"

    n_files = len(tasks)
    n_unique_patients = combined["patient_id"].nunique()
    assert n_unique_patients == n_files, (
        f"Expected {n_files} unique patients (one per file), got {n_unique_patients} "
        "-- a file may have been silently dropped or a patient_id collided."
    )

    print(f"Combined dataset: {len(combined):,} rows across {n_unique_patients:,} patients.")
    for hosp, g in combined.groupby("hospital"):
        n_pat = g["patient_id"].nunique()
        n_septic_pat = g.loc[g[config.LABEL_COL] == 1, "patient_id"].nunique()
        print(f"  Hospital {hosp}: {n_pat:,} patients, {n_septic_pat:,} ever-septic "
              f"({100 * n_septic_pat / n_pat:.1f}%), {len(g):,} rows, "
              f"{100 * g[config.LABEL_COL].mean():.2f}% positive rows.")

    config.RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(config.RAW_PARQUET, engine="pyarrow", index=False)
    print(f"Saved to {config.RAW_PARQUET}")


if __name__ == "__main__":
    main()
