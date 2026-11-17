"""Build the retrieval arm's neighbour folders: for each VAL pool scan, the 5 TRAIN patients whose CTs are closest by
MerlinOnc pan-cancer ep99 embedding (cosine), each with its CT, its five answers, its report, its rank and its
similarity; plus the free 5-nearest-neighbour vote gate (MerlinOnc and Pillar0). Spec:
planning/vista_eval_vlm/2026-11-17-ct-findings-retrieval-arm.md (Approved 2026-11-17). The layout copies the EHR
agent's scripts/build_probe_tool.py: durable files on the mount (inside the secure environment), aggregates printed,
checks before anything is used, never an id or a row printed.

Neighbours come from the whole TRAIN split of vista_bench v1.7 ``per_ct_diagnostic_tasks`` (every train scan with a
label for at least one of the five questions and a per-CT embedding), at most one scan per neighbour patient, the 5
closest regardless of their findings, one set shared by all five questions. A candidate scan needs exactly one
report text (the one its labels were read from, joined on accession number as build_data.py ``reports``) and, to be
picked for a VAL scan, a volume. No VAL or TEST-sample patient is ever a candidate.

The vote: for each question, the mean of the neighbours' yes/no answers over those that have one ("not scorable"
skipped); with none, the question's TRAIN yes rate. Two variants are scored: ``shared`` (the 5 the agent sees: the
gate) and ``per_finding`` (the 5 closest patients with an answer for that question). The TRAIN leave-one-patient-out
vote (one scan per TRAIN patient as the query, neighbours from the other TRAIN patients; volumes not checked) gives
the per-question vote AUROC and the similarity ranges the agent is told; no VAL label is used for either.

    select   load embeddings, TRAIN candidates and report counts; pick neighbours (MerlinOnc) for VAL-200; score the
             vote gate for MerlinOnc and Pillar0 on VAL-200 and on TRAIN leave-one-out; write the gate table, the
             vote scores and the private picks. Exit 1 (STOP) if the MerlinOnc shared-vote mean is under 0.60.
    volumes  convert every picked neighbour scan to neighbour_scans/<id>/{ct.npz, meta.json} (build_data.py's
             converter). Heavy (up to ~1,000 scans, ~13 s each, ~5 GB peak per worker): run on phil-hcpu.
    folders  fetch the picked scans' reports and write neighbours/<VAL id>/{README.md, neighbours.json,
             n1..n5/{answers.json, report.txt}}. CTs are not copied here; agentic.py stages them per session.
    check    every check the spec names (200 folders, 5 neighbours each, TRAIN only, one scan per neighbour patient,
             no VAL or TEST-sample patient, answers equal the table, similarities recomputed, a brute-force re-pick of
             20 VAL scans, the vote file reproduced from the folders, every neighbour volume present). Prints PASS.

    cd optimizer/ct-findings-5
    OMP_NUM_THREADS=4 uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with numpy --with scikit-learn python -B scripts/build_neighbours.py select
    OMP_NUM_THREADS=1 nice -n 19 uv run -q --no-project --with nibabel --with numpy python -B scripts/build_neighbours.py volumes --workers 8
    uv run -q --no-project --with google-cloud-bigquery python -B scripts/build_neighbours.py folders
    OMP_NUM_THREADS=4 uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with numpy python -B scripts/build_neighbours.py check

BigQuery runs on this host's application-default login. Prints counts only: never an id, a UID, a date or a row.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as bd  # noqa: E402  (the label query, ids, NIfTI paths, the converter, the report table)

QUESTIONS = bd.QUESTIONS
KEYS = tuple(k for k, _t in QUESTIONS)
EMB_DIR = Path("/mnt/su-vista-uscentral1/vistabench/embeddings/v1_7/ct/per_ct_diagnostic_tasks")
#: (name shown, embedding stem). The first one picks the neighbours (spec, Decided: MerlinOnc pan-cancer ep99;
#: Pillar0 is measured in the gate, and switched to only on Phil's word).
MODELS = (("MerlinOnc pan-cancer ep99", "merlinonc_top_centered_pancancer_postdx_ep99"), ("Pillar0", "pillar0"))
K = 5  # DECIDE: spec, the 5 closest
GATE_MEAN = 0.60  # DECIDE: spec, stop before any paid step if the MerlinOnc vote mean is below this
NEIGH_DIR = bd.DATA_DIR / "neighbours"
SCAN_DIR = bd.DATA_DIR / "neighbour_scans"
BUILD = bd.DATA_DIR / "neighbours_build.json"  # aggregates: the gate table, TRAIN leave-one-out, similarity ranges
VOTES = bd.DATA_DIR / "neighbours_vote_val.json"  # per VAL id vote scores, for score_baselines.py
PICKS = bd.PRIVATE_DIR / "neighbour_picks.json"  # per VAL id: the picks with patient, accession and NIfTI
SCAN_MAP = bd.DATA_DIR / "neighbours_scan_map.json"  # per VAL id: its neighbours' opaque ids, n1 first (agentic.py)
SEED = "ct-findings-5-v1:neighbours:loo-query"
WORD = {1: "yes", 0: "no", None: "not scorable"}
BRUTE_FORCE_SAMPLE = 20


def scan_key_of_nifti(path: str) -> str:
    return Path(path).name.removesuffix(".nii.gz")


def load_embeddings(stem: str):
    """{study__series: row} and the L2-normalised float32 matrix. Stops on a repeated scan, a ragged or zero row."""
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(EMB_DIR / f"{stem}_embeddings.parquet", columns=["image_study_uid", "image_series_uid", "embedding"])
    col = t.column("embedding").combine_chunks()
    lens = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    dim = int(lens[0])
    if (lens != dim).any():
        raise SystemExit(f"{stem}: embeddings of different lengths")
    mat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, dim)
    keys = [f"{a}__{b}" for a, b in zip(t.column("image_study_uid").to_pylist(), t.column("image_series_uid").to_pylist())]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{stem}: one scan has two embedding rows")
    norms = np.linalg.norm(mat, axis=1)
    if not np.isfinite(mat).all() or (norms == 0).any():
        raise SystemExit(f"{stem}: a non-finite or all-zero embedding")
    mat /= norms[:, None]
    return {k: i for i, k in enumerate(keys)}, mat


def report_counts(client, accessions: list[str]) -> dict[str, int]:
    """Per accession, how many distinct non-blank report texts (the fetch in ``folders`` keeps exactly one)."""
    from google.cloud import bigquery

    sql = f"""
        select accession_number, count(distinct if(trim(report_note_text) = '', null, report_note_text)) as n
        from `{bd.REPORT_TABLE}`
        where accession_number in unnest(@acc)
        group by accession_number"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("acc", "STRING", sorted(set(accessions)))]))
    return {r["accession_number"]: int(r["n"]) for r in job.result()}


def fetch_reports(client, accessions: list[str]) -> dict[str, list[str]]:
    from google.cloud import bigquery

    sql = f"""
        select accession_number, array_agg(distinct report_note_text ignore nulls) as texts
        from `{bd.REPORT_TABLE}`
        where accession_number in unnest(@acc)
        group by accession_number"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("acc", "STRING", sorted(set(accessions)))]))
    return {r["accession_number"]: [t for t in r["texts"] if t and t.strip()] for r in job.result()}


def the_split(client):
    """(TRAIN scans with a label: {key, person_id, accession, study, series, labels}, valid-split patients, every
    labelled train/valid scan's labels by key). From build_data.load_scans (the table's train and valid rows only)."""
    scans = bd.load_scans(client)
    train = [{"key": k, **s} for k, s in sorted(scans.items()) if s["split"] == "train"]
    val_patients = {s["person_id"] for s in scans.values() if s["split"] == "val"}
    return train, val_patients, {k: s["labels"] for k, s in scans.items()}


def val_pool():
    """[(VAL id, scan key)] for the VAL pool, and its gold rows (splits/val.jsonl, what the scorer uses)."""
    pool = json.loads((bd.DATA_DIR / "pools.json").read_text())["val"]
    id_map = {m["id"]: m for m in map(json.loads, (bd.PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    gold = {r["id"]: r for r in map(json.loads, (bd.DATA_DIR / "splits" / "val.jsonl").read_text().splitlines()) if r}
    if any(id_map[i]["split"] != "val" for i in pool):
        raise SystemExit("a VAL pool id is not a VAL scan in the id map")
    return [(i, scan_key_of_nifti(id_map[i]["nifti"])) for i in sorted(pool)], gold, id_map


def walk(order, cands, k: int, *, exclude_patient=None, eligible=None, need_key=None) -> list[int]:
    """The first ``k`` candidates in ``order`` (closest first), one per patient, skipping ``exclude_patient``, the
    ones ``eligible`` refuses and, with ``need_key``, the ones with no answer for that question."""
    out, used = [], set()
    for i in order:
        c = cands[i]
        p = c["person_id"]
        if p in used or p == exclude_patient:
            continue
        if need_key is not None and c["labels"][need_key] is None:
            continue
        if eligible is not None and not eligible(i):
            continue
        used.add(p)
        out.append(int(i))
        if len(out) == k:
            break
    return out


def vote(neigh: list[int], cands, key: str, rate: float) -> float:
    said = [cands[i]["labels"][key] for i in neigh if cands[i]["labels"][key] is not None]
    return sum(said) / len(said) if said else rate


def auroc(gold: list[int], score: list[float]) -> float | None:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(gold, score)) if 0 < sum(gold) < len(gold) else None


def pcts(values) -> dict:
    import numpy as np

    v = np.asarray(values, dtype=float)
    return {f"p{q}": round(float(np.percentile(v, q)), 4) for q in (10, 50, 90)}


def cmd_select(_args) -> int:
    import numpy as np
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=bd.PROJECT)
    train, val_patients, _labels = the_split(client)
    pool, gold, id_map = val_pool()
    pool_patients = {int(id_map[i]["person_id"]) for i, _k in pool}
    test_patients = set(pd.read_parquet(bd.TEST_SAMPLE, columns=["person_id"])["person_id"].astype(int))
    if {s["person_id"] for s in train} & (val_patients | pool_patients | test_patients):
        raise SystemExit("a TRAIN patient is also a VAL or TEST-sample patient: the split rule is broken")
    counts = report_counts(client, [s["accession"] for s in train if s["accession"]])
    stats = Counter(train_scans=len(train), train_patients=len({s["person_id"] for s in train}))
    with_report = []
    for s in train:
        n = counts.get(s["accession"], 0) if s["accession"] else 0
        if n != 1:
            stats["dropped_no_single_report"] += 1
            continue
        with_report.append(s)
    rates = {k: float(np.mean([s["labels"][k] for s in with_report if s["labels"][k] is not None])) for k in KEYS}
    have_volume: dict[str, bool] = {}
    build = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "k": K, "gate_mean": GATE_MEAN,
             "train_yes_rate": {k: round(v, 4) for k, v in rates.items()}, "models": {}}
    votes_out: dict = {"_what": "5-NN vote per VAL pool scan and question: mean of the neighbours' yes/no answers "
                                "(not scorable skipped; none -> the TRAIN yes rate)", "k": K}
    picks_out: dict = {}
    for model_i, (name, stem) in enumerate(MODELS):
        index, mat = load_embeddings(stem)
        cands = [s for s in with_report if s["key"] in index]
        cmat = mat[[index[s["key"]] for s in cands]]
        m_stats = {"candidates": len(cands), "candidate_patients": len({s["person_id"] for s in cands}),
                   "dropped_no_embedding": len(with_report) - len(cands)}
        missing_q = [i for i, key in pool if key not in index]
        if missing_q:
            raise SystemExit(f"{name}: {len(missing_q)} VAL pool scans have no embedding")
        qmat = mat[[index[key] for _i, key in pool]]
        del mat

        def eligible(ci: int) -> bool:
            c = cands[ci]
            if c["key"] not in have_volume:
                have_volume[c["key"]] = bd.nifti(c["study"], c["series"]).is_file()
            return have_volume[c["key"]]

        # VAL-200: the shared set (what the agent sees) and the per-question variant.
        shared, per_finding, val_rows, sim1, sim5 = {}, {}, {}, [], []
        sims_all = qmat @ cmat.T
        for qi, (vid, _key) in enumerate(pool):
            sims = sims_all[qi]
            order = np.argsort(-sims, kind="stable")
            neigh = walk(order, cands, K, eligible=eligible)
            if len(neigh) < K:
                raise SystemExit(f"{name}: a VAL scan has fewer than {K} eligible neighbours")
            sim1.append(float(sims[neigh[0]]))
            sim5.append(float(sims[neigh[-1]]))
            shared[vid] = {k: vote(neigh, cands, k, rates[k]) for k in KEYS}
            per_finding[vid] = {k: vote(walk(order, cands, K, eligible=eligible, need_key=k), cands, k, rates[k])
                                for k in KEYS}
            val_rows[vid] = neigh
            if model_i == 0:
                picks_out[vid] = [{"rank": r, "similarity": round(float(sims[ci]), 6),
                                   "nid": bd.opaque_id(cands[ci]["person_id"], cands[ci]["study"], cands[ci]["series"]),
                                   "person_id": cands[ci]["person_id"], "accession": cands[ci]["accession"],
                                   "nifti": str(bd.nifti(cands[ci]["study"], cands[ci]["series"])),
                                   "labels": cands[ci]["labels"]} for r, ci in enumerate(neigh, 1)]
        del sims_all
        val = {}
        for k in KEYS:
            ids = [vid for vid, _key in pool if gold[vid][k] is not None]
            g = [gold[v][k] for v in ids]
            no_said = sum(all(cands[ci]["labels"][k] is None for ci in val_rows[v]) for v in ids)
            val[k] = {"n": len(ids), "yes": sum(g), "auroc_shared": auroc(g, [shared[v][k] for v in ids]),
                      "auroc_per_finding": auroc(g, [per_finding[v][k] for v in ids]),
                      "scans_with_no_neighbour_answer": no_said}
        mean_shared = float(np.mean([val[k]["auroc_shared"] for k in KEYS]))
        mean_pf = float(np.mean([val[k]["auroc_per_finding"] for k in KEYS]))

        # TRAIN leave-one-patient-out: one scan per TRAIN patient as the query; volumes not checked.
        rng = random.Random(SEED)
        by_patient: dict[int, list[int]] = {}
        for ci, c in enumerate(cands):
            by_patient.setdefault(c["person_id"], []).append(ci)
        queries = [rng.choice(sorted(by_patient[p])) for p in sorted(by_patient)]
        loo_gold = {k: [] for k in KEYS}
        loo_score = {k: [] for k in KEYS}
        tsim1, tsim5 = [], []
        for start in range(0, len(queries), 500):
            block = queries[start:start + 500]
            sims_block = cmat[block] @ cmat.T
            for bi, ci in enumerate(block):
                sims = sims_block[bi]
                order = np.argsort(-sims, kind="stable")
                neigh = walk(order, cands, K, exclude_patient=cands[ci]["person_id"])
                tsim1.append(float(sims[neigh[0]]))
                tsim5.append(float(sims[neigh[-1]]))
                for k in KEYS:
                    if cands[ci]["labels"][k] is not None:
                        loo_gold[k].append(cands[ci]["labels"][k])
                        loo_score[k].append(vote(neigh, cands, k, rates[k]))
        loo = {k: {"n": len(loo_gold[k]), "yes": int(sum(loo_gold[k])), "auroc_shared": auroc(loo_gold[k], loo_score[k])}
               for k in KEYS}
        build["models"][name] = {
            "stem": stem, **m_stats,
            "val_pool": {"per_question": val, "mean_auroc_shared": mean_shared, "mean_auroc_per_finding": mean_pf},
            "train_leave_one_out": {"queries": len(queries), "per_question": loo,
                                    "mean_auroc_shared": float(np.mean([loo[k]["auroc_shared"] for k in KEYS]))},
            "similarity": {"val_rank1": pcts(sim1), "val_rank5": pcts(sim5),
                           "train_rank1": pcts(tsim1), "train_rank5": pcts(tsim5)}}
        votes_out[name] = {"shared": shared, "per_finding": per_finding}
        print(f"{name}: {len(cands)} TRAIN candidate scans ({m_stats['candidate_patients']} patients); "
              f"TRAIN leave-one-out over {len(queries)} patients")
        print(f"  {'question':<20}{'VAL n':>6}{'yes':>5}{'shared':>9}{'per-q':>8}{'TRAIN LOO':>11}")
        for k in KEYS:
            v, t = val[k], loo[k]
            print(f"  {k:<20}{v['n']:>6}{v['yes']:>5}{v['auroc_shared']:>9.3f}{v['auroc_per_finding']:>8.3f}"
                  f"{t['auroc_shared']:>11.3f}")
        print(f"  {'mean':<31}{mean_shared:>9.3f}{mean_pf:>8.3f}"
              f"{build['models'][name]['train_leave_one_out']['mean_auroc_shared']:>11.3f}")
        s = build["models"][name]["similarity"]
        print(f"  similarity (10th/50th/90th pct): VAL rank 1 {s['val_rank1']}, rank 5 {s['val_rank5']}; "
              f"TRAIN rank 1 {s['train_rank1']}, rank 5 {s['train_rank5']}")
    build["counts"] = {"dropped_no_single_report": 0, **stats, "volumes_checked": len(have_volume),
                       "volumes_missing": sum(not v for v in have_volume.values()),
                       "picked_scans_distinct": len({p["nid"] for ps in picks_out.values() for p in ps}),
                       "picked_patients_distinct": len({p["person_id"] for ps in picks_out.values() for p in ps})}
    gate_value = build["models"][MODELS[0][0]]["val_pool"]["mean_auroc_shared"]
    build["gate"] = {"model": MODELS[0][0], "mean_auroc_shared": gate_value, "threshold": GATE_MEAN,
                     "passed": gate_value >= GATE_MEAN}
    BUILD.write_text(json.dumps(build, indent=1), encoding="utf-8")
    VOTES.write_text(json.dumps(votes_out), encoding="utf-8")
    PICKS.write_text(json.dumps(picks_out), encoding="utf-8")
    c = build["counts"]
    print(f"TRAIN: {c['train_scans']} labelled scans ({c['train_patients']} patients), {c['dropped_no_single_report']} "
          f"without exactly one report; volumes checked {c['volumes_checked']}, missing {c['volumes_missing']}; "
          f"picked {c['picked_scans_distinct']} distinct scans of {c['picked_patients_distinct']} patients")
    print(f"written {BUILD.name}, {VOTES.name} (mount) and the private picks")
    if not build["gate"]["passed"]:
        print(f"STOP: the MerlinOnc 5-NN vote mean is {gate_value:.3f}, under {GATE_MEAN}: no paid step (spec)")
        return 1
    print(f"GATE PASS: MerlinOnc 5-NN vote mean {gate_value:.3f} >= {GATE_MEAN}")
    return 0


def picked_scans() -> dict[str, dict]:
    picks = json.loads(PICKS.read_text())
    out = {}
    for ps in picks.values():
        for p in ps:
            out[p["nid"]] = p
    return out


def cmd_volumes(args) -> int:
    scans = picked_scans()
    wanted = sorted(scans)
    if args.limit:
        wanted = wanted[: args.limit]
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    status, errors = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        items = [(n, scans[n]["nifti"], str(SCAN_DIR)) for n in wanted]
        for i, (st, detail) in enumerate(pool.map(bd.convert, items), 1):
            status[st] += 1
            if detail:
                errors[detail] += 1
            if i % 50 == 0:
                print(f"{i}/{len(wanted)} {dict(status)}", flush=True)
    print(f"done {len(wanted)}: {dict(status)}; errors by type {dict(errors)}")
    return 1 if status["failed"] else 0


def readme(build: dict) -> str:
    m = build["models"][MODELS[0][0]]
    s = m["similarity"]
    loo = m["train_leave_one_out"]["per_question"]
    aucs = "; ".join(f"{k} {loo[k]['auroc_shared']:.2f}" for k in KEYS)
    return (
        "# Similar patients\n\n"
        f"The folders n1 to n{K} hold {K} patients from a separate training set (never the patient you are reading). "
        "They were picked because their CT scans are the closest to this scan by MerlinOnc, a CT foundation model: "
        "the cosine similarity of the two scans' whole-scan embeddings. Their findings played no part in picking "
        "them. n1 is the closest.\n\n"
        "Each folder holds:\n"
        "- `ct/`: that patient's CT, in the same format as your scan (ct.npz and meta.json; the same viewer works)\n"
        "- `answers.json`: the answer key's five answers for that scan: \"yes\", \"no\", or \"not scorable\" where "
        "the report did not let the question be answered\n"
        "- `report.txt`: the radiologist's report the answers were read from\n\n"
        "`neighbours.json` lists each folder's rank and similarity score.\n\n"
        f"How close is close: across the training set, the closest patient's similarity is usually between "
        f"{s['train_rank1']['p10']:.2f} and {s['train_rank1']['p90']:.2f} (median {s['train_rank1']['p50']:.2f}), "
        f"and the {K}th closest's between {s['train_rank5']['p10']:.2f} and {s['train_rank5']['p90']:.2f} (median "
        f"{s['train_rank5']['p50']:.2f}).\n\n"
        f"How much the neighbours' answers say on their own: on the training set, ranking each patient by the share of "
        f"their {K} neighbours answered yes (each patient left out of their own neighbours) gives these AUROCs "
        f"(0.5 is chance, 1 is perfect): {aucs}.\n"
    )


def cmd_folders(_args) -> int:
    from google.cloud import bigquery

    build = json.loads(BUILD.read_text())
    if not build["gate"]["passed"]:
        raise SystemExit("the vote gate did not pass: no folders (spec)")
    picks = json.loads(PICKS.read_text())
    scans = picked_scans()
    texts = fetch_reports(bigquery.Client(project=bd.PROJECT), [p["accession"] for p in scans.values()])
    bad = [n for n, p in scans.items() if len(texts.get(p["accession"], [])) != 1]
    if bad:
        raise SystemExit(f"{len(bad)} picked scans do not have exactly one report now: nothing written")
    text = readme(build)
    NEIGH_DIR.mkdir(parents=True, exist_ok=True)
    for vid, ps in sorted(picks.items()):
        d = NEIGH_DIR / vid
        d.mkdir(exist_ok=True)
        (d / "README.md").write_text(text, encoding="utf-8")
        rows = []
        for p in ps:
            nd = d / f"n{p['rank']}"
            nd.mkdir(exist_ok=True)
            answers = {k: WORD[p["labels"][k]] for k in KEYS}
            (nd / "answers.json").write_text(json.dumps(answers, indent=1) + "\n", encoding="utf-8")
            (nd / "report.txt").write_text(texts[p["accession"]][0].strip() + "\n", encoding="utf-8")
            rows.append({"folder": f"n{p['rank']}", "rank": p["rank"], "similarity": round(p["similarity"], 4),
                         "answers": answers})
        (d / "neighbours.json").write_text(json.dumps({
            "picked_by": "MerlinOnc pan-cancer CT embedding, cosine similarity of whole scans, closest first",
            "neighbours": rows}, indent=1) + "\n", encoding="utf-8")
    SCAN_MAP.write_text(json.dumps({vid: [p["nid"] for p in ps] for vid, ps in sorted(picks.items())}), encoding="utf-8")
    print(f"written {len(picks)} folders under {NEIGH_DIR} ({len(scans)} distinct neighbour scans' reports) and "
          f"{SCAN_MAP.name}")
    return 0


def cmd_check(_args) -> int:
    import pandas as pd
    from google.cloud import bigquery

    fails = 0

    def report(what: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {what}{': ' + detail if detail else ''}")

    build = json.loads(BUILD.read_text())
    picks = json.loads(PICKS.read_text())
    votes = json.loads(VOTES.read_text())
    pool, _gold, id_map = val_pool()
    pool_ids = [i for i, _k in pool]
    train, val_patients, labels = the_split(bigquery.Client(project=bd.PROJECT))
    train_patients = {s["person_id"] for s in train}
    pool_patients = {int(id_map[i]["person_id"]) for i in pool_ids}
    test_patients = set(pd.read_parquet(bd.TEST_SAMPLE, columns=["person_id"])["person_id"].astype(int))
    every = [p for ps in picks.values() for p in ps]
    print("neighbour picks and folders")
    report("the vote gate passed", build["gate"]["passed"], f"MerlinOnc shared-vote mean {build['gate']['mean_auroc_shared']:.3f}")
    report("one pick list per VAL pool scan", sorted(picks) == pool_ids, f"{len(picks)} lists, {len(pool_ids)} VAL scans")
    report(f"{K} neighbours each, ranks 1..{K}, similarity not increasing",
           all([p["rank"] for p in ps] == list(range(1, K + 1)) and
               all(a["similarity"] >= b["similarity"] for a, b in zip(ps, ps[1:])) for ps in picks.values()))
    report("at most one scan per neighbour patient in each list",
           all(len({p["person_id"] for p in ps}) == len(ps) for ps in picks.values()))
    report("every neighbour is a TRAIN-split patient", all(p["person_id"] in train_patients for p in every))
    report("no neighbour is a valid-split or VAL-pool patient",
           not any(p["person_id"] in val_patients or p["person_id"] in pool_patients for p in every),
           f"{len({p['person_id'] for p in every})} neighbour patients")
    report("no neighbour is a patient of the VLMs' TEST sample", not any(p["person_id"] in test_patients for p in every))
    report("no neighbour id is a VAL id", not ({p["nid"] for p in every} & set(pool_ids)))
    report("every neighbour's answers equal the table's labels",
           all(labels.get(scan_key_of_nifti(p["nifti"])) == p["labels"] for p in every))

    # Similarities recomputed from the embedding file, and a brute-force re-pick of a sample of VAL scans.
    index, mat = load_embeddings(MODELS[0][1])
    key_of = {i: k for i, k in pool}
    diffs = [abs(float(mat[index[key_of[v]]] @ mat[index[scan_key_of_nifti(p["nifti"])]]) - p["similarity"])
             for v, ps in picks.items() for p in ps]
    report("similarities recomputed from the embeddings", max(diffs) < 1e-4, f"largest difference {max(diffs):.2e}")
    with_emb = [s for s in train if s["key"] in index]
    counts = report_counts(bigquery.Client(project=bd.PROJECT), [s["accession"] for s in with_emb if s["accession"]])
    cand = pd.DataFrame([{"key": s["key"], "person_id": s["person_id"]} for s in with_emb
                         if s["accession"] and counts.get(s["accession"], 0) == 1])
    cmat = mat[[index[k] for k in cand["key"]]]
    sample = random.Random("brute-force").sample(pool_ids, BRUTE_FORCE_SAMPLE)
    agree = 0
    for v in sample:
        sims = cmat @ mat[index[key_of[v]]]
        ranked = cand.assign(sim=sims).sort_values("sim", ascending=False)
        best, seen = [], set()
        for key, person in zip(ranked["key"], ranked["person_id"]):  # one per patient, a volume required
            if person in seen or not (bd.NIFTI_ROOT / f"{key}.nii.gz").is_file():
                continue
            seen.add(person)
            best.append(key)
            if len(best) == K:
                break
        agree += best == [scan_key_of_nifti(p["nifti"]) for p in picks[v]]
    report(f"a plain sort re-picks the same {K} for {BRUTE_FORCE_SAMPLE} sampled VAL scans", agree == len(sample),
           f"{agree} of {len(sample)} match")

    # The folders the sessions get, the vote reproduced from them, and the volumes.
    rates = build["train_yes_rate"]
    folder_ok, vote_ok, text_ok, vol_ok, map_ok = 0, 0, 0, 0, 0
    scan_map = json.loads(SCAN_MAP.read_text()) if SCAN_MAP.is_file() else {}
    back = {"yes": 1, "no": 0, "not scorable": None}
    for v in pool_ids:
        d = NEIGH_DIR / v
        names = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
        if names != sorted(["README.md", "neighbours.json", *[f"n{r}" for r in range(1, K + 1)]]):
            continue
        folder_ok += 1
        said = [json.loads((d / f"n{r}" / "answers.json").read_text()) for r in range(1, K + 1)]
        text_ok += all((d / f"n{r}" / "report.txt").stat().st_size > 20 for r in range(1, K + 1))
        want = votes[MODELS[0][0]]["shared"][v]
        got = {}
        for k in KEYS:
            ys = [back[s[k]] for s in said if back[s[k]] is not None]
            got[k] = sum(ys) / len(ys) if ys else rates[k]
        vote_ok += all(abs(got[k] - want[k]) < 1e-3 for k in KEYS)
        vol_ok += all((SCAN_DIR / p["nid"] / n).is_file() for p in picks[v] for n in ("ct.npz", "meta.json"))
        map_ok += scan_map.get(v) == [p["nid"] for p in picks[v]]
    report("every VAL scan has README.md, neighbours.json and n1..n5", folder_ok == len(pool_ids), f"{folder_ok} of {len(pool_ids)}")
    report("every neighbour has a report", text_ok == len(pool_ids), f"{text_ok} of {len(pool_ids)} folders")
    report("the vote file reproduces from the folders' answers", vote_ok == len(pool_ids), f"{vote_ok} of {len(pool_ids)}")
    report("the staging map names each folder's neighbours in rank order", map_ok == len(pool_ids),
           f"{map_ok} of {len(pool_ids)}")
    report("every neighbour volume is converted", vol_ok == len(pool_ids),
           f"{vol_ok} of {len(pool_ids)} folders ({len({p['nid'] for p in every})} distinct scans)")
    print("PASS: neighbour folders ready" if not fails else f"FAIL: {fails} check(s) failed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    v = sub.add_parser("volumes")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--limit", type=int, default=0)
    sub.add_parser("folders")
    sub.add_parser("check")
    args = ap.parse_args()
    return {"select": cmd_select, "volumes": cmd_volumes, "folders": cmd_folders, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
