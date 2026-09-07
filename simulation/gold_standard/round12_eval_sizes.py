"""Round-12: mean evaluation-half sizes (no borrowing / final design) for
every randomized-boundary configuration, for the row-wise Table 8.

The round-10 artifact stores the two mean evaluation-half sizes for its
four configurations; the round-11 artifact added the two location-held
mixings but stored only the per-split size GAP.  This script re-runs all
six configurations with the identical RNG (20260908) and records, per
configuration, the mean evaluation-half size of each design, asserting
that the estimates and gaps reproduce the stored artifacts exactly.

Output: artifacts_round4/calibration/round12_eval_sizes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation" / "gold_standard"))
from round10_randomized_crossfit import (  # noqa: E402
    SHIFTS, normal_weights, rand_threshold, expected_reject)

RV = ROOT / "artifacts_round4" / "validation" / "pvals"
CAL = ROOT / "artifacts_round4" / "calibration" / "mixture_calibration.json"
R10 = ROOT / "artifacts_round4" / "calibration" / "round10_randomized_crossfit.json"
R11 = ROOT / "artifacts_round4" / "calibration" / "round11_inference_scope.json"
OUT = ROOT / "artifacts_round4" / "calibration" / "round12_eval_sizes.json"
R = 200


def main() -> int:
    Wcan = {float(k): v
            for k, v in json.loads(CAL.read_text())["mixture_weights"].items()}
    nul, alt = {}, {}
    for m in ("weak_only", "cap50"):
        nul[m] = {s: np.load(RV / f"null_s{s}_f1.0.npz")[m] for s in SHIFTS}
        alt[m] = {s: np.load(RV / f"alt_s{s}_f1.0.npz")[m] for s in SHIFTS}
    n = len(nul["weak_only"][0.0])
    half = n // 2
    r10 = json.loads(R10.read_text())["runs"]
    r11 = json.loads(R11.read_text())["runs"]

    configs = [("canonical", Wcan, 0.02513), ("canonical", Wcan, 0.025),
               ("N(0,1)", normal_weights(0.0, 1.0), 0.02513),
               ("N(0,0.5)", normal_weights(0.0, 0.5), 0.02513),
               ("N(+0.124,1)", normal_weights(0.124, 1.0), 0.02513),
               ("N(+0.124,0.5)", normal_weights(0.124, 0.5), 0.02513)]
    res = {"R": R, "runs": {}}
    for name, W, tgt in configs:
        rng = np.random.default_rng(20260908)
        ests, sw, sm = [], [], []
        for _ in range(R):
            perm = {s: rng.permutation(n) for s in SHIFTS}
            h0 = {s: perm[s][:half] for s in SHIFTS}
            h1 = {s: perm[s][half:] for s in SHIFTS}
            fv, fw, fm = [], [], []
            for a, b in ((h0, h1), (h1, h0)):
                c_w, g_w = rand_threshold(
                    {s: nul["weak_only"][s][a[s]] for s in SHIFTS}, W, tgt)
                c_m, g_m = rand_threshold(
                    {s: nul["cap50"][s][a[s]] for s in SHIFTS}, W, tgt)
                fv.append(expected_reject(alt["cap50"], b, W, c_m, g_m)
                          - expected_reject(alt["weak_only"], b, W, c_w, g_w))
                fw.append(expected_reject(nul["weak_only"], b, W, c_w, g_w))
                fm.append(expected_reject(nul["cap50"], b, W, c_m, g_m))
            ests.append(float(np.mean(fv)))
            sw.append(float(np.mean(fw)))
            sm.append(float(np.mean(fm)))
        e, sw, sm = np.array(ests), np.array(sw), np.array(sm)
        key = f"{name}@{tgt}"
        entry = {"mean": round(float(e.mean()), 5),
                 "eval_size_weak_mean": round(float(sw.mean()), 5),
                 "eval_size_cap50_mean": round(float(sm.mean()), 5),
                 "eval_size_gap_mean": round(float((sm - sw).mean()), 6),
                 "frac_positive": round(float((e > 0).mean()), 4)}
        assert entry["mean"] == r11[key]["est"]["mean"], key
        assert entry["eval_size_gap_mean"] == r11[key]["gap"]["mean"], key
        if key in r10:
            assert entry["eval_size_weak_mean"] == r10[key]["eval_size_weak_mean"], key
            assert entry["eval_size_cap50_mean"] == r10[key]["eval_size_cap50_mean"], key
        res["runs"][key] = entry
        print(key, entry, flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
