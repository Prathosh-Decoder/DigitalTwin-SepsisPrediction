#!/usr/bin/env python3
"""Example usage for the packaged six-hour sepsis predictor."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patient_file", type=Path, help="PhysioNet-style PSV or CSV patient history")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "sepsis_next_6h_predictor.pkl",
    )
    args = parser.parse_args()

    with args.model.open("rb") as handle:
        predictor = pickle.load(handle)
    print(json.dumps(predictor.predict_patient(args.patient_file), indent=2))


if __name__ == "__main__":
    main()
