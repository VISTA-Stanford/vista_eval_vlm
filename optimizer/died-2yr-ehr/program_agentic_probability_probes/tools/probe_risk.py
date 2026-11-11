"""Print a patient's probe risks (probes.json) as a short table.

    python3 -B probe_risk.py <path to probes.json>

For each probe and horizon: this patient's score (or not available), the probe's TRAIN AUROC and coverage, the
calibration band the score falls in and how many patients in that band died.
"""

import json
import sys


def band_of(score, bands):
    for b in bands:
        if score <= b["score_to"]:
            return b
    return bands[-1]


def main(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(data["about"], "\n")
    for p in data["probes"]:
        print(f"{p['probe']}: {p['reads']}")
        for horizon, h in p["horizons"].items():
            head = f"  death within {horizon}: AUROC {h['train_auroc']:.3f}, covers {h['coverage_percent']:.0f}% of patients"
            if not h["available"]:
                print(f"{head}; this patient: not available")
                continue
            b = band_of(h["this_patient_score"], h["calibration"])
            print(f"{head}; this patient's score {h['this_patient_score']:.4f} is in the band "
                  f"{b['score_from']:.4f}-{b['score_to']:.4f}, where {b['died_percent']:.1f}% of "
                  f"{b['patients']} patients died")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
