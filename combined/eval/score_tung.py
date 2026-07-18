"""Score TUNG's model on the same val + test patients, per ICU hour.

Run with the dedicated Tung venv:  ../.venv-tung/bin/python score_tung.py
(xgboost only -- NO lightgbm import in this process). Uses the reconstructed TungModel
(see combined/tung_predictor.py) because the packaged pickle's own predict methods
segfault in this environment.

Output: combined/eval/results/tung_scores.parquet
    patient_id, ICULOS, tung_prob, tung_raw
"""
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "sepsis_pipeline"
DATA_A = ROOT / "training_setA" / "training_setA"
DATA_B = ROOT / "training_setB" / "training_setB"
OUT = Path(__file__).resolve().parent / "results" / "tung_scores.parquet"

sys.path.insert(0, str(ROOT / "combined"))
from tung_predictor import TungModel  # noqa: E402


def psv_path(pid: int) -> Path:
    fname = f"p{int(pid):06d}.psv"
    a = DATA_A / fname
    return a if a.exists() else DATA_B / fname


def main() -> None:
    model = TungModel()
    val_ids = json.loads((PIPE / "artifacts" / "val_patient_ids.json").read_text())
    test_ids = json.loads((PIPE / "artifacts" / "test_patient_ids.json").read_text())
    patient_ids = sorted(set(val_ids) | set(test_ids))
    print(f"Scoring {len(patient_ids):,} val+test patients through Tung's model...")

    frames = []
    t0 = time.time()
    for i, pid in enumerate(patient_ids, 1):
        traj = model.trajectory(pd.read_csv(psv_path(pid), sep="|"))
        frames.append(pd.DataFrame({
            "patient_id": pid,
            "ICULOS": traj["ICULOS"].to_numpy(),
            "tung_prob": traj["tung_prob"].to_numpy(),
            "tung_raw": traj["tung_raw"].to_numpy(),
        }))
        if i % 1000 == 0:
            print(f"  {i:,}/{len(patient_ids):,}  ({time.time() - t0:.0f}s)")

    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(out):,} rows) in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
