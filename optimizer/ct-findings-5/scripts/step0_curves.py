"""Step 0 of the CT context-library idea (assessment page 2026-11-21, approved; idea note
planning/vista_eval_vlm/ideas-ct-context-library-test-time-compute.md): the free CPU bars any exemplar-based agent arm
on ct-findings-5 has to clear, on the same VAL-200 pool as every agent pass.

    knn      the k-nearest-neighbour label vote for k = 5, 10, 25, 50, 100 (MerlinOnc ep99 picks; Pillar0 as the
             ablation). Same candidates and walk as the retrieval arm (build_neighbours.py: TRAIN, exactly one report,
             volume on disk, one scan per patient), so k=5 must reproduce its gate table (MerlinOnc 0.663).
    probe    a linear probe trained on only n positive + n negative TRAIN patients per question (n = 5..250, many
             random draws; regularisation picked by cross-validation inside those 2n patients only), and on all
             candidates. This is the same-number-of-labels bar for an agent given 2n labelled examples.
    fusion   late fusion (equal-weight mean of within-question rank percentiles; nothing tuned on VAL) of the
             retrieval-arm agent and the untuned agent passes with the MerlinOnc full-TRAIN probe (published
             per-example scores), with the 5-NN vote and with the n=50 probe; paired bootstrap of fused minus probe.
             Phil 2026-11-21: the late fusion of k-NN/probe + agent is the bar to beat.
    --kb-arms  fusion for the knowledge-base pilot (spec 2026-11-23): its library and empty-library VAL arms beside arm D,
             with paired intervals over the two target questions (lymph nodes, pneumonitis) alone too; written to
             ct-findings-5-v1_7-runs/knowledge-base/fusion.json.
    cite     how many TRAIN candidate reports cite a series and/or image number (counted inside BigQuery; no report
             text leaves it). Bounds a slice-level library built from report citations.

    cd optimizer/ct-findings-5
    OMP_NUM_THREADS=4 uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with numpy --with scikit-learn python -B scripts/step0_curves.py

Prints and writes aggregates only (never an id, accession, date or text) to
ct-findings-5-v1_7-runs/context-library-step0/step0.json on the mount.
"""

from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as bd  # noqa: E402
import build_neighbours as bn  # noqa: E402

KEYS = bn.KEYS
KS = (5, 10, 25, 50, 100)
NS = (5, 10, 25, 50, 100, 250)  # per class
DRAWS = {5: 30, 10: 30, 25: 20, 50: 20, 100: 10, 250: 5}
FUSION_N = 50
BOOT = 2000
RUNS = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil/ct-findings-5-v1_7-runs")
OUT = RUNS / "context-library-step0" / "step0.json"
AGENT_PASSES = {"retrieval arm": RUNS / "retrieval-arm" / "retrieval.json",
                **{f"untuned {p}": RUNS / "retrieval-arm" / "untuned" / f"{p}.json" for p in ("prob-r1", "prob-r2", "prob-r4")}}
# --example-arms (spec 2026-11-22-ct-findings-example-ct-arms.md): fuse the three example-CT arms beside the 11-17 arm,
# each scored only on the scans it answers (arm C: the 190 with organ ranges), written beside the arms' answers; the
# report-citation count is step 0's alone and is skipped.
EXAMPLE_ARMS = {"retrieval arm": RUNS / "retrieval-arm" / "retrieval.json",
                **{f"arm {a}": RUNS / "example-ct-arms" / f"arm-{a.lower()}.json" for a in "ABCD"}}
EXAMPLE_OUT = RUNS / "example-ct-arms" / "fusion.json"
# --kb-arms (spec 2026-11-23-ct-knowledge-base-pilot.md): the knowledge-base library arm and its empty-library arm, with
# arm D (the best image arm so far) beside them, each on the scans it answers (the 190). The win is judged on the two
# target questions, so the paired intervals are also given over those two alone.
KB_ARMS = {"kb library": RUNS / "knowledge-base" / "val-library.json", "kb empty library": RUNS / "knowledge-base" / "val-empty.json",
           "arm D": RUNS / "example-ct-arms" / "arm-d.json"}
KB_OUT = RUNS / "knowledge-base" / "fusion.json"
KB_TARGETS = ("lesion_lymph_nodes", "pneumonitis")


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def pct(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 4), "p10": round(float(np.percentile(v, 10)), 4),
            "p90": round(float(np.percentile(v, 90)), 4), "draws": int(len(v))}


def ranks(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def main() -> int:
    from google.cloud import bigquery

    global AGENT_PASSES, OUT
    example_arms = "--example-arms" in sys.argv[1:]
    kb_arms = "--kb-arms" in sys.argv[1:]
    if example_arms:
        AGENT_PASSES, OUT = EXAMPLE_ARMS, EXAMPLE_OUT
    if kb_arms:
        AGENT_PASSES, OUT, example_arms = KB_ARMS, KB_OUT, True  # the same per-arm scoring; citations skipped

    client = bigquery.Client(project=bd.PROJECT)
    train, _val_patients, _ = bn.the_split(client)
    pool, gold, id_map = bn.val_pool()
    vids = [v for v, _k in pool]
    counts = bn.report_counts(client, [s["accession"] for s in train if s["accession"]])
    with_report = [s for s in train if s["accession"] and counts.get(s["accession"], 0) == 1]
    with ThreadPoolExecutor(32) as ex:
        have = list(ex.map(lambda s: bd.nifti(s["study"], s["series"]).is_file(), with_report))
    pool_cands = [s for s, h in zip(with_report, have) if h]
    rates = {k: float(np.mean([s["labels"][k] for s in with_report if s["labels"][k] is not None])) for k in KEYS}
    out = {"what": "step 0 bars, ct-findings-5 VAL-200 (see docstring)", "val_scans": len(vids),
           "train_with_one_report": len(with_report), "train_with_volume": len(pool_cands),
           "train_patients_with_volume": len({s["person_id"] for s in pool_cands}), "knn": {}, "probe": {}}
    print(f"TRAIN candidates: {len(pool_cands)} scans with one report and a volume "
          f"({out['train_patients_with_volume']} patients); VAL pool {len(vids)}")
    gold_k = {k: [v for v in vids if gold[v][k] is not None] for k in KEYS}

    # ---- knn curve -------------------------------------------------------------------------------------------
    vote_scores = {}
    emb = {}
    for name, stem in bn.MODELS:
        index, mat = bn.load_embeddings(stem)
        cands = [s for s in pool_cands if s["key"] in index]
        cmat = mat[[index[s["key"]] for s in cands]]
        qmat = mat[[index[key] for _v, key in pool]]
        emb[name] = (cands, cmat, qmat)
        sims = qmat @ cmat.T
        orders = [np.argsort(-sims[i], kind="stable") for i in range(len(vids))]
        res = {}
        for k in KS:
            sc = {v: {q: bn.vote(bn.walk(orders[i], cands, k), cands, q, rates[q]) for q in KEYS}
                  for i, v in enumerate(vids)}
            per_q = {q: auroc([gold[v][q] for v in gold_k[q]], [sc[v][q] for v in gold_k[q]]) for q in KEYS}
            res[k] = {"per_question": {q: round(a, 4) for q, a in per_q.items()},
                      "mean": round(float(np.mean(list(per_q.values()))), 4)}
            if name == bn.MODELS[0][0] and k == 5:
                vote_scores = sc
        out["knn"][name] = res
        print(f"k-NN vote, {name}: " + ", ".join(f"k={k} {res[k]['mean']:.3f}" for k in KS))
        del mat, sims

    # ---- probe curve -----------------------------------------------------------------------------------------
    probe_all = {}  # per question: this script's full-TRAIN probe (the published file misses some VAL scans)
    probe50 = {}  # per question: mean VAL score over the n=50 draws, for the fusion bar
    for name, _stem in bn.MODELS:
        cands, cmat, qmat = emb[name]
        res = {}
        for n in NS + ("all",):
            draws = 1 if n == "all" else DRAWS[n]
            means, perq, capped = [], {q: [] for q in KEYS}, set()
            acc50 = {q: [] for q in KEYS}  # one rank array per draw: fused draw by draw, never averaged first
            for d in range(draws):
                rng = np.random.default_rng(1000 * (0 if n == "all" else n) + d)
                aucs = []
                for q in KEYS:
                    # one scan per patient, labelled for q
                    by_p = {}
                    for i in rng.permutation(len(cands)):
                        c = cands[i]
                        if c["labels"][q] is not None and c["person_id"] not in by_p:
                            by_p[c["person_id"]] = int(i)
                    pos = [i for i in by_p.values() if cands[i]["labels"][q] == 1]
                    neg = [i for i in by_p.values() if cands[i]["labels"][q] == 0]
                    if n == "all":
                        idx = pos + neg
                        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], cv=5,
                                                                                    max_iter=3000, scoring="roc_auc"))
                    else:
                        npos, nneg = min(n, len(pos)), min(n, len(neg))
                        if (npos, nneg) != (n, n):
                            capped.add(f"{q}: {npos}+{nneg}")
                        idx = list(rng.choice(pos, npos, replace=False)) + list(rng.choice(neg, nneg, replace=False))
                        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
                            Cs=[1e-3, 1e-2, 1e-1, 1], cv=min(5, n), max_iter=3000, scoring="roc_auc"))
                    y = np.array([cands[i]["labels"][q] for i in idx])
                    clf.fit(cmat[idx], y)
                    s = clf.predict_proba(qmat)[:, 1]
                    sel = [vids.index(v) for v in gold_k[q]]
                    a = auroc([gold[v][q] for v in gold_k[q]], s[sel])
                    perq[q].append(a)
                    aucs.append(a)
                    if n == FUSION_N and name == bn.MODELS[0][0]:
                        acc50[q].append(ranks(s))
                    if n == "all" and name == bn.MODELS[0][0]:
                        probe_all[q] = ranks(s)
                means.append(float(np.mean(aucs)))
            if n == FUSION_N and name == bn.MODELS[0][0]:
                probe50 = acc50
            res[str(n)] = {"mean": pct(means), "per_question": {q: round(float(np.mean(v)), 4) for q, v in perq.items()},
                           "train_patients_per_question": "all" if n == "all" else 2 * n,
                           "capped_to_available": sorted(capped)}
            print(f"probe, {name}, n={n}{'/class' if n != 'all' else ''}: mean {res[str(n)]['mean']['mean']:.3f} "
                  f"[{res[str(n)]['mean']['p10']:.3f}, {res[str(n)]['mean']['p90']:.3f}] over {draws} draws"
                  + (f"; capped {sorted(capped)}" if capped else ""))
        out["probe"][name] = res

    # ---- fusion ----------------------------------------------------------------------------------------------
    probe_full = probe_all
    vote_r = {q: ranks([vote_scores[v][q] for v in vids]) for q in KEYS}
    rng = np.random.default_rng(0)
    out["fusion"] = {}
    for pname, path in AGENT_PASSES.items():
        ans = json.loads(path.read_text())["answers"]
        # Every row is ranked, scored and resampled over the scans this pass answers (all of vids except for arm C,
        # which answers the 190 with organ ranges; spec: no failed-answer fill for the scans it never got).
        pv = [v for v in vids if v in ans]
        idx = np.array([vids.index(v) for v in pv])
        pf = {q: ranks(probe_full[q][idx]) for q in KEYS}
        vr = {q: ranks(vote_r[q][idx]) for q in KEYS}
        p50 = {q: [ranks(r[idx]) for r in probe50[q]] for q in KEYS}
        failed = 0
        agent_r = {}
        for q in KEYS:
            raw = []
            for v in pv:
                a = ans.get(v, {})
                x = a.get(q) if not a.get("error") else None
                if not isinstance(x, (int, float)):
                    failed += 1
                    x = 0.5  # the task scorer's failed-answer rule
                raw.append(float(x))
            agent_r[q] = ranks(raw)
        rows = {"agent": agent_r, "MerlinOnc probe (full TRAIN)": pf, "MerlinOnc 5-NN vote": vr,
                f"MerlinOnc probe (n={FUSION_N}/class)": p50,
                "agent + probe (full)": {q: (agent_r[q] + pf[q]) / 2 for q in KEYS},
                "agent + 5-NN vote": {q: (agent_r[q] + vr[q]) / 2 for q in KEYS},
                f"agent + probe (n={FUSION_N}/class)": {q: [(agent_r[q] + r) / 2 for r in p50[q]] for q in KEYS},
                "probe (full) + 5-NN vote": {q: (pf[q] + vr[q]) / 2 for q in KEYS}}
        pos_of = {v: j for j, v in enumerate(pv)}
        sel = {q: np.array([pos_of[v] for v in gold_k[q] if v in pos_of]) for q in KEYS}
        g = {q: np.array([gold[v][q] for v in gold_k[q] if v in pos_of]) for q in KEYS}
        def sc_auc(scq, q, rows_i=None):
            arrs = scq if isinstance(scq, list) else [scq]
            vals = []
            for a in arrs:
                a = a[sel[q]]
                if rows_i is None:
                    vals.append(auroc(g[q], a))
                else:
                    vals.append(roc_auc_score(g[q][rows_i], a[rows_i]))
            return float(np.mean(vals))

        table = {}
        for rname, sc in rows.items():
            pq_ = {q: sc_auc(sc[q], q) for q in KEYS}
            table[rname] = {"mean": round(float(np.mean(list(pq_.values()))), 4),
                            "per_question": {q: round(a, 4) for q, a in pq_.items()}}

        def boot(a_name, b_name, nboot=BOOT, keys=KEYS):
            diffs = []
            n = len(pv)
            while len(diffs) < nboot:
                i = rng.integers(0, n, n)
                ds = []
                for q in keys:
                    keep = np.isin(sel[q], i)
                    # resample scans: map each drawn VAL index to its row in this question
                    pos = {int(x): j for j, x in enumerate(sel[q])}
                    rows_i = [pos[int(x)] for x in i if int(x) in pos]
                    y = g[q][rows_i]
                    if not 0 < y.sum() < len(y):
                        break
                    ds.append(sc_auc(rows[a_name][q], q, rows_i) - sc_auc(rows[b_name][q], q, rows_i))
                    del keep
                else:
                    diffs.append(np.mean(ds))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            return [round(float(lo), 4), round(float(hi), 4)]

        table["paired: agent+probe(full) minus probe(full)"] = boot("agent + probe (full)", "MerlinOnc probe (full TRAIN)")
        table[f"paired: agent+probe(n={FUSION_N}) minus probe(n={FUSION_N})"] = boot(
            f"agent + probe (n={FUSION_N}/class)", f"MerlinOnc probe (n={FUSION_N}/class)", nboot=500)
        if kb_arms:
            for rname in ("agent", "MerlinOnc probe (full TRAIN)", "MerlinOnc 5-NN vote", "agent + probe (full)",
                          "agent + 5-NN vote"):
                table[rname]["targets_mean"] = round(float(np.mean([table[rname]["per_question"][q] for q in KB_TARGETS])), 4)
            table["paired targets: agent minus probe(full)"] = boot("agent", "MerlinOnc probe (full TRAIN)", keys=KB_TARGETS)
            table["paired targets: agent minus 5-NN vote"] = boot("agent", "MerlinOnc 5-NN vote", keys=KB_TARGETS)
            table["paired targets: agent+probe(full) minus probe(full)"] = boot("agent + probe (full)",
                                                                                "MerlinOnc probe (full TRAIN)", keys=KB_TARGETS)
        table["failed_answers"] = failed
        table["scans"] = len(pv)
        out["fusion"][pname] = table
        print(f"fusion, {pname} on {len(pv)} scans (failed answers {failed}):")
        for rname in rows:
            print(f"  {rname:<34}{table[rname]['mean']:.3f}")
        for k_ in [k for k in table if k.startswith("paired")]:
            print(f"  {k_}: 95% {table[k_]}")

    if example_arms:
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"written {OUT}")
        return 0

    # ---- report citations ------------------------------------------------------------------------------------
    sql = f"""
      with r as (
        select accession_number, any_value(lower(report_note_text)) t
        from `{bd.REPORT_TABLE}` where accession_number in unnest(@acc) and trim(report_note_text) != ''
        group by accession_number)
      select count(*) n,
        countif(regexp_contains(t, r'\\bseries\\s*#?\\s*\\d+')) series_no,
        countif(regexp_contains(t, r'\\b(image|img|im)\\s*#?\\s*\\d+')) image_no,
        countif(regexp_contains(t, r'\\b(series|se)\\s*#?\\s*\\d+[^.]{{0,25}}\\b(image|img|im)\\s*#?\\s*\\d+')) series_then_image,
        countif(regexp_contains(t, r'\\(\\s*\\d+\\s*[:/]\\s*\\d+\\s*\\)')) paren_pair,
        countif(regexp_contains(t, r'\\(\\s*\\d+\\s*[:/]\\s*([6-9]\\d|\\d{{3,}})\\s*\\)')) paren_pair_second_over_59,
        countif(regexp_contains(t, r'\\b(series|se)\\s*#?\\s*\\d+[^.]{{0,25}}\\b(image|img|im)\\s*#?\\s*\\d+')
                or regexp_contains(t, r'\\(\\s*\\d+\\s*[:/]\\s*([6-9]\\d|\\d{{3,}})\\s*\\)')) any_slice_ref_strict
      from r"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("acc", "STRING", sorted({s["accession"] for s in pool_cands}))]))
    r = dict(next(iter(job.result())).items())
    out["cite"] = {k: int(v) for k, v in r.items()}
    n = max(r["n"], 1)
    print(f"report citations over {r['n']} TRAIN reports: " + ", ".join(
        f"{k} {v} ({v/n:.0%})" for k, v in r.items() if k != "n"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
