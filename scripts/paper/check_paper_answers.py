"""Check the exported answer tables against what the paper's figure scripts read today (paper-code stage 2).

  1. EHR death agents (Sonnet v5 / v0, Haiku ladder): every patient's two probabilities equal the agents-vs-embeddings
     loader's (results/agents-vs-our-models-subgroups/step10 load_start + step25's v0 read), the reader Fig 2b uses.
  2. Over-time agents (Haiku v5 / v0, Sonnet zero-shot, Sonnet restaging-tuned): the same decision-point rows as
     the figure readers (step7 answers(), step25 read()) both ways, and every probability equal.
  3. CT death agent v4 / v0: mean AUROC over the two horizons on all test patients equals the saved
     fig5-vlm-agent/data/agent_ct_death_r2_test_scores.csv (patients_set all_test, MACRO).
  4. General LLM runs: each exported table re-scored with evalset.score.score_rows equals the original answers file
     scored the same way (AUROC or balanced accuracy per question, 200 draws).
Prints aggregates only. Exit 0 when every check passes.

  cd <vista_eval_vlm checkout> && UV_PYTHON=3.11 uv run python scripts/paper/check_paper_answers.py [DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evalset import paper_export as PE
from evalset.score import prepare, score_rows

RES = Path(
    "/mnt/su-vista-uscentral1/chaudhari_lab/phil/planning/vista_bench/paper/results"
)
AVS = RES / "agents-vs-our-models-subgroups"
COLS = [
    "step",
    "recipe",
    "arm",
    "question_id",
    "person_id",
    "label",
    "pred",
    "refused",
    "score",
    "yesno",
]


def table(root: Path, manifest: dict, key: str) -> pd.DataFrame:
    return pd.read_parquet(root / manifest[key]["run_id"] / "answers.parquet")


def check_ehr(root, man) -> bool:
    sys.path.insert(0, str(AVS))
    import step10_numbers_table_and_disagreement as S10

    _, sc, _ = S10.load_start()
    idm = {
        json.loads(x)["id"]: str(json.loads(x)["person_id"])
        for x in Path(S10.L8.IDMAP).read_text().splitlines()
    }
    v0 = json.loads(
        (
            Path(S10.L8.M)
            / "died-2yr-ehr-v1_7-private-test-sonnet/sonnet-program-v0.test-test-program-v0-test-dfa15c5d.answers.json"
        ).read_text()
    )["answers"]
    sc["SU"] = {idm[i]: [S10.L8.prob(v, k) for k in S10.L8.KEYS] for i, v in v0.items()}
    pairs = {
        "ehr_sonnet_v5": "SA",
        "ehr_sonnet_v0": "SU",
        "ehr_haiku_A_v0": "A v0",
        "ehr_haiku_A_v2": "A v2",
        "ehr_haiku_R_v0": "R v0",
        "ehr_haiku_R_v8": "R v8",
        "ehr_haiku_L_v0": "L v0",
    }
    ok = True
    for key, ref in pairs.items():
        d = table(root, man, key).pivot(
            index="person_id", columns="task", values="probability"
        )
        r = pd.DataFrame(
            sc[ref], index=["died_any_cause_1_yr", "died_any_cause_2_yr"]
        ).T
        r.index = r.index.astype(str)
        same = d.index.sort_values().equals(r.index.sort_values()) and np.allclose(
            d.loc[r.index].to_numpy(), r.to_numpy()
        )
        ok &= same
        print(
            f"1 {key} vs loader '{ref}': {len(d):,} patients, {'identical' if same else 'DIFFERENT'}"
        )
    return ok


def check_over_time(root, man) -> bool:
    """Each over-time table against the figure readers: Fig 3a's step7 answers() for the five-point runs (pointed at
    the Sonnet zero-shot folder for that run) and step25's read() for the restaging-only tuned Sonnet run. The
    decision-point sets must be equal both ways (test rows of points.parquet), and every probability equal."""
    sys.path.insert(0, str(AVS))
    import step7_x4_x9_tuned_agent as S7
    import step25_sonnet_tuned_restaging as S25

    runs = Path(S7.PRIV["test"]) / "runs"
    pts = pd.read_parquet(Path(S7.PRIV["test"]) / "points.parquet")
    test_rows = set(pts.loc[pts.split == "test", "row"].astype(str))
    zs = S7.ZEROSHOT["test"]
    ok = True
    for key, which, folder in (
        ("over_time_haiku_v5", "climbed", None),
        ("over_time_haiku_v0", "zeroshot", None),
        (
            "over_time_sonnet_v0",
            "zeroshot",
            runs / "over-time-test-sonnet-zeroshot-r1/answers",
        ),
        (
            "over_time_sonnet_restaging_v8",
            "restaging",
            runs / "over-time-test-sonnet-restaging-tuned-r1/answers",
        ),
    ):
        if which == "restaging":
            ref, _ = S25.read(folder)
        else:
            S7.ZEROSHOT["test"] = str(folder) if folder else zs
            ref = S7.answers("test", which)
        S7.ZEROSHOT["test"] = zs
        want = {k for k in ref if k in test_rows}
        d = table(root, man, key)
        got = set(d.case_id)
        bad = 0
        for r in d.itertuples():
            k = (
                "died_within_1_year"
                if r.task.endswith("1_yr")
                else "died_within_2_years"
            )
            bad += abs(S7.prob(ref.get(r.case_id), k) - r.probability) > 1e-12
        same = got == want and bad == 0 and len(got) > 0
        ok &= same
        print(
            f"2 {key} vs {'step25 read' if which == 'restaging' else 'step7 ' + which}: export {len(got):,} / reader {len(want):,} "
            f"decision-point rows, {len(got ^ want)} differ in membership, {bad} probabilities differ, "
            f"{int(d.label.isna().sum())} rows without a label (kept; the readers drop them)"
        )
    return ok


def check_ct(root, man) -> bool:
    ref = pd.read_csv(RES / "fig5-vlm-agent/data/agent_ct_death_r2_test_scores.csv")
    ref = ref[(ref.patients_set == "all_test") & (ref.question_id == "MACRO")]
    ok = True
    for key in ("ct_death_v4", "ct_death_v0"):
        d = table(root, man, key)
        auc = float(
            np.mean([_auroc(g.label, g.probability) for _, g in d.groupby("task")])
        )
        rec = ref[ref.recipe.str.endswith(key[-2:])]
        want = float(rec.auroc.iloc[0]) if len(rec) else float("nan")
        same = abs(auc - want) < 1e-9
        ok &= same
        print(
            f"3 {key}: mean AUROC on all test {auc:.4f} vs saved {want:.4f} {'PASS' if same else 'FAIL'}"
        )
    return ok


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(np.asarray(y, int), np.asarray(s, float))


def check_llm(root, man) -> bool:
    ok, n = True, 0
    for key, m in man.items():
        if m["role"] != "llm":
            continue
        src = PE.EVALSET_RUNS / (key[len("llm/") :] + ".csv")
        a = prepare(pd.read_csv(src, dtype={"case_id": str, "person_id": str}))
        a = a[a.split == "test"]
        d = table(root, man, key)
        yesno = d.probability.isna().all()
        b = d.assign(
            recipe=d.program_version,
            arm=a.arm.iloc[0],
            score=d.probability,
            yesno=yesno,
            pred=[{"yes": 1, "no": 0}.get(x) for x in d.answer] if yesno else None,
        )
        ta, tb = score_rows(a[COLS], n_boot=200), score_rows(b[COLS], n_boot=200)
        col = "balanced_accuracy" if yesno else "auroc"
        x, y = ta.set_index("question_id")[col], tb.set_index("question_id")[col]
        same = x.index.equals(y.index) and np.allclose(
            x.to_numpy(float), y.to_numpy(float), equal_nan=True
        )
        ok &= same
        n += 1
        if not same:
            print(f"4 {key}: DIFFERENT {col}")
    print(
        f"4 general LLM runs: {n} re-scored, {'all identical' if ok else 'some differ'}"
    )
    return ok


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else PE.OUT_ROOT
    man = json.loads((root / "MANIFEST.json").read_text())
    ok = [
        check_ehr(root, man),
        check_over_time(root, man),
        check_ct(root, man),
        check_llm(root, man),
    ]
    print("PASS" if all(ok) else "FAIL")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
