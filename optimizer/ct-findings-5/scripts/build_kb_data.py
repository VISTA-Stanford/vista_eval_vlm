"""Data for the CT knowledge-base pilot (spec planning/vista_eval_vlm/2026-11-23-ct-knowledge-base-pilot.md, Approved
2026-11-23): the tester's TRAIN rounds, the builder's TRAIN source scans, and what each session is staged with.

Two TRAIN sets, from patients that never overlap (spec, Decided):
    tester   500 scans = 5 rounds of 100, each half positive for lymph nodes or pneumonitis. The TRAIN pool's scans
             with organ ranges and an embedding, topped up with new TRAIN scans (pneumonitis positives first, then
             lymph-node positives, then others) to reach 500 with 250 positives and at least 75 pneumonitis positives.
    source   300 scans the builder may read (report, answers, CT, organ ranges) and cut entries from: every eligible
             pneumonitis positive among the neighbour scans plus new ones up to 80, then 110 lymph-node positives and
             110 others from the neighbour scans.
Neither set holds a VAL or TEST-sample patient, nor a patient of the other set; one scan per patient.

    select      BigQuery labels + local files -> kb/splits/kb_train.jsonl (tester gold), kb/rounds.json,
                kb/source.json; private kb_id_map.jsonl. Prints counts only.
    volumes     convert the new scans' NIfTIs into patients/<id>/ (build_data.convert)
    reports     each source scan's report -> kb/source_reports/<id>.txt (TRAIN only)
    organs      organ ranges -> kb/organs/<id>.json for tester and source scans (build_organ_ranges.ranges)
    similarity  MerlinOnc whole-scan cosine to each source scan -> kb/similarity/<id>.json, for tester scans and the
                VAL scans with organ ranges (the 190 every arm is scored on)
    stage DEST  the builder's read-only copy of the source scans, on local disk (the containers cannot see the mount)
    check       patients disjoint, rounds as drawn, every file present, no identifier in staged text. Prints PASS.

    cd optimizer/ct-findings-5
    uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with nibabel --with numpy \\
        python -B scripts/build_kb_data.py select

Counts only: never an id, a date, a UID or report text.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as bd  # noqa: E402
import build_organ_ranges as bor  # noqa: E402

KB = bd.DATA_DIR / "kb"
KB_ID_MAP = bd.PRIVATE_DIR / "kb_id_map.jsonl"
SEED = "ct-findings-5-kb-v1"
ROUNDS, PER_ROUND = 5, 100
TESTER_SIZE, TESTER_POSITIVES, TESTER_PNEUMONITIS = ROUNDS * PER_ROUND, ROUNDS * PER_ROUND // 2, 75
SOURCE_PNEUMONITIS, SOURCE_NODES, SOURCE_OTHER = 80, 110, 110
#: The library is sorted by build_neighbours.MODELS[0], the MerlinOnc pan-cancer ep99 embedding the neighbours used.
TARGETS = ("lesion_lymph_nodes", "pneumonitis")


def klass(labels: dict) -> str:
    """P: pneumonitis positive; N: lymph-node positive (not pneumonitis); O: everything else."""
    if labels.get("pneumonitis") == 1:
        return "P"
    if labels.get("lesion_lymph_nodes") == 1:
        return "N"
    return "O"


def deal(ids_by_class: dict[str, list[str]], rounds: int, rng: random.Random) -> list[list[str]]:
    """Shuffle each class and deal it round-robin, so every round gets the same mix (within one scan per class)."""
    out: list[list[str]] = [[] for _ in range(rounds)]
    k = 0
    for c in sorted(ids_by_class):
        ids = sorted(ids_by_class[c])
        rng.shuffle(ids)
        for i in ids:
            out[k % rounds].append(i)
            k += 1
    return [sorted(r) for r in out]


def pick_tester(pool: list[dict], new: list[dict], rng: random.Random) -> list[dict]:
    """The TRAIN pool's eligible scans, then new scans by class, to the targets above. Raises when short."""
    chosen = list(pool)
    by = Counter(klass(s["labels"]) for s in chosen)
    cands = {c: [s for s in new if klass(s["labels"]) == c] for c in "PNO"}
    for c in cands.values():
        rng.shuffle(c)
    need_p = max(0, TESTER_PNEUMONITIS - by["P"])
    need_n = max(0, TESTER_POSITIVES - by["P"] - need_p - by["N"])
    if need_p > len(cands["P"]) or need_n > len(cands["N"]):
        raise SystemExit(f"not enough new TRAIN positives: need P {need_p} N {need_n}, have {len(cands['P'])} {len(cands['N'])}")
    chosen += cands["P"][:need_p] + cands["N"][:need_n]
    others = [s for s in chosen if klass(s["labels"]) == "O"]
    room = TESTER_SIZE - len(chosen)
    if room < 0:
        # More pool "others" than room: drop the extra others at random (never a positive).
        drop = {s["id"] for s in rng.sample(others, -room)}
        chosen = [s for s in chosen if s["id"] not in drop]
    elif room > len(cands["O"]):
        raise SystemExit(f"not enough new TRAIN scans to fill the tester set: need {room}")
    else:
        chosen += cands["O"][:room]
    if len(chosen) != TESTER_SIZE:
        raise SystemExit(f"tester set has {len(chosen)} scans, not {TESTER_SIZE}")
    return chosen


def pick_source(neigh: list[dict], new: list[dict], rng: random.Random) -> list[dict]:
    by = {c: [s for s in neigh if klass(s["labels"]) == c] for c in "PNO"}
    for c in by.values():
        rng.shuffle(c)
    newp = [s for s in new if klass(s["labels"]) == "P"]
    rng.shuffle(newp)
    p = by["P"][:SOURCE_PNEUMONITIS]
    p += newp[: SOURCE_PNEUMONITIS - len(p)]
    # Others: explicit negatives on both targets first, so the builder has clean "no" examples.
    clean = [s for s in by["O"] if all(s["labels"].get(t) == 0 for t in TARGETS)]
    rest = [s for s in by["O"] if s not in clean]
    o = (clean + rest)[:SOURCE_OTHER]
    n = by["N"][:SOURCE_NODES]
    if len(p) < SOURCE_PNEUMONITIS or len(n) < SOURCE_NODES or len(o) < SOURCE_OTHER:
        raise SystemExit(f"source short: P {len(p)} N {len(n)} O {len(o)}")
    return p + n + o


def cmd_select(_args) -> int:
    import pandas as pd
    from google.cloud import bigquery

    import build_neighbours as bn

    rng = random.Random(SEED)
    scans = bd.load_scans(bigquery.Client(project=bd.PROJECT))
    test_patients = set(pd.read_parquet(bd.TEST_SAMPLE, columns=["person_id"])["person_id"].astype(int))
    val_patients = {s["person_id"] for s in scans.values() if s["split"] == "val"}
    covered = set()
    import csv
    with bor.TABLE.open(newline="") as f:
        for r in csv.DictReader(f):
            covered.add(Path(r["nifti_filename"]).name)
    index, _mat = bn.load_embeddings(bn.MODELS[0][1])

    def eligible(key: str, s: dict) -> bool:
        return (s["split"] == "train" and s["person_id"] not in test_patients and s["person_id"] not in val_patients
                and f"{key}.nii.gz" in covered and key in index)

    def row(key: str, s: dict, volume: str) -> dict:
        return {"id": bd.opaque_id(s["person_id"], s["study"], s["series"]), "key": key, "person_id": s["person_id"],
                "accession": s["accession"], "labels": s["labels"], "volume": volume}

    pool_ids = set(json.loads((bd.DATA_DIR / "pools.json").read_text())["train"])
    by_id = {bd.opaque_id(s["person_id"], s["study"], s["series"]): (k, s) for k, s in scans.items()}
    pool = [row(*by_id[i], "patients") for i in sorted(pool_ids) if eligible(*by_id[i])]
    pool_patients = {by_id[i][1]["person_id"] for i in pool_ids}
    picks = json.loads((bd.PRIVATE_DIR / "neighbour_picks.json").read_text())
    neigh_keys = sorted({Path(p["nifti"]).name.removesuffix(".nii.gz") for ps in picks.values() for p in ps})
    neigh_patients = {scans[k]["person_id"] for k in neigh_keys if k in scans}

    # New scans: one per patient, from patients in neither the pool nor the neighbour picks.
    per_patient: dict[int, list[str]] = {}
    for k, s in scans.items():
        if eligible(k, s) and s["person_id"] not in pool_patients and s["person_id"] not in neigh_patients:
            per_patient.setdefault(s["person_id"], []).append(k)
    # The patient's pneumonitis-positive scan if it has one, else a lymph-node positive, else any: patients have
    # about six TRAIN scans each, so a random scan would mostly miss the rare positives the pilot needs.
    def one_scan(ks: list[str]) -> str:
        for c in "PN":
            hit = sorted(k for k in ks if klass(scans[k]["labels"]) == c)
            if hit:
                return rng.choice(hit)
        return rng.choice(sorted(ks))

    new = [row(k, scans[k], "patients") for k in (one_scan(ks) for _p, ks in sorted(per_patient.items()))]
    rng.shuffle(new)
    half = len(new) // 2  # tester and source draw new patients from separate halves: disjoint by construction
    tester = pick_tester(pool, new[:half], rng)
    tester_patients = {s["person_id"] for s in tester}

    seen: set[int] = set()
    neigh = []
    for k in neigh_keys:
        s = scans.get(k)
        if s and eligible(k, s) and s["person_id"] not in tester_patients and s["person_id"] not in seen:
            seen.add(s["person_id"])
            neigh.append(row(k, s, "neighbour_scans"))
    source = pick_source(neigh, new[half:], rng)

    rounds = deal({c: [s["id"] for s in tester if klass(s["labels"]) == c] for c in "PNO"}, ROUNDS, rng)
    KB.mkdir(parents=True, exist_ok=True)
    (KB / "splits").mkdir(exist_ok=True)
    keys = [q for q, _t in bd.QUESTIONS]
    with (KB / "splits" / "kb_train.jsonl").open("w") as f:
        for s in sorted(tester, key=lambda s: s["id"]):
            f.write(json.dumps({"id": s["id"], **{k: s["labels"].get(k) for k in keys}}) + "\n")
    draw = {"seed": SEED, "tester": dict(Counter(klass(s["labels"]) for s in tester)),
            "tester_from_pool": len(pool), "source": dict(Counter(klass(s["labels"]) for s in source)),
            "source_from_neighbours": sum(s["volume"] == "neighbour_scans" for s in source),
            "per_round": [dict(Counter(klass(next(t["labels"] for t in tester if t["id"] == i)) for i in r)) for r in rounds]}
    (KB / "rounds.json").write_text(json.dumps({**{str(n): r for n, r in enumerate(rounds, 1)}, "_draw": draw}, indent=1))
    (KB / "source.json").write_text(json.dumps([{"id": s["id"], "answers": {k: s["labels"].get(k) for k in keys},
                                                  "volume": s["volume"]} for s in sorted(source, key=lambda s: s["id"])], indent=1))
    with KB_ID_MAP.open("w") as f:
        for role, items in (("tester", tester), ("source", source)):
            for s in items:
                f.write(json.dumps({"id": s["id"], "role": role, "person_id": s["person_id"], "accession": s["accession"],
                                    "nifti": str(bd.nifti(*s["key"].split("__"))), "key": s["key"], "volume": s["volume"]}) + "\n")
    print(json.dumps(draw, indent=1))
    return 0


def id_map() -> list[dict]:
    return [json.loads(line) for line in KB_ID_MAP.read_text().splitlines() if line.strip()]


def volume_dir(m: dict) -> Path:
    return bd.DATA_DIR / m["volume"] / m["id"]


def cmd_volumes(args) -> int:
    missing = [m for m in id_map() if not (volume_dir(m) / "meta.json").is_file()]
    if any(m["volume"] != "patients" for m in missing):
        raise SystemExit("a neighbour scan has no volume: build_neighbours.py volumes first")
    todo = [(m["id"], m["nifti"]) for m in missing]
    status = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (st, _d) in enumerate(pool.map(bd.convert, todo), 1):
            status[st] += 1
            if n % 25 == 0:
                print(f"{n}/{len(todo)} {dict(status)}", flush=True)
    print(f"done {len(todo)} to convert: {dict(status)}")
    return 1 if status["failed"] else 0


def cmd_reports(_args) -> int:
    from google.cloud import bigquery

    src = [m for m in id_map() if m["role"] == "source"]
    sql = f"""
        select accession_number, array_agg(distinct report_note_text ignore nulls) as texts
        from `{bd.REPORT_TABLE}` where accession_number in unnest(@acc) group by accession_number"""
    job = bigquery.Client(project=bd.PROJECT).query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("acc", "STRING", sorted({m["accession"] for m in src}))]))
    texts = {r["accession_number"]: [t for t in r["texts"] if t and t.strip()] for r in job.result()}
    out = KB / "source_reports"
    out.mkdir(parents=True, exist_ok=True)
    status = Counter()
    for m in src:
        found = texts.get(m["accession"], [])
        if len(found) != 1:
            status["no_report" if not found else "several_reports"] += 1
            (out / f"{m['id']}.txt").unlink(missing_ok=True)
            continue
        (out / f"{m['id']}.txt").write_text(found[0].strip() + "\n", encoding="utf-8")
        status["written"] += 1
    print(f"source scans {len(src)}: {dict(status)}")
    return 0


def cmd_organs(_args) -> int:
    maps = id_map()
    table = bor.load_table({Path(m["nifti"]).name for m in maps})
    out = KB / "organs"
    out.mkdir(parents=True, exist_ok=True)
    status = Counter()
    for m in maps:
        r = bor.ranges(table.get(Path(m["nifti"]).name), volume_dir(m) / "meta.json")
        status["available" if r["available"] else "not_available"] += 1
        (out / f"{m['id']}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"organ ranges: {dict(status)}")
    return 0


def val_scored_ids() -> list[str]:
    """The VAL scans every arm with organ ranges is scored on (run.py val-pass's filter)."""
    pools = json.loads((bd.DATA_DIR / "pools.json").read_text())
    return sorted(i for i in pools["val"] if (bd.DATA_DIR / "organ_ranges" / i / "target.json").is_file())


def cmd_similarity(_args) -> int:
    import build_neighbours as bn

    index, mat = bn.load_embeddings(bn.MODELS[0][1])
    maps = id_map()
    src = [m for m in maps if m["role"] == "source"]
    S = mat[[index[m["key"]] for m in src]]
    val_map = {m["id"]: Path(m["nifti"]).name.removesuffix(".nii.gz")
               for m in map(json.loads, (bd.PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    targets = [(m["id"], m["key"]) for m in maps if m["role"] == "tester"] + [(i, val_map[i]) for i in val_scored_ids()]
    out = KB / "similarity"
    out.mkdir(parents=True, exist_ok=True)
    for tid, key in targets:
        sims = S @ mat[index[key]]
        (out / f"{tid}.json").write_text(json.dumps({m["id"]: round(float(v), 4) for m, v in zip(src, sims)}))
    print(f"similarity files: {len(targets)} ({len(targets) - len(val_scored_ids())} tester, {len(val_scored_ids())} VAL), "
          f"each over {len(src)} source scans")
    return 0


SOURCE_README = """# Source scans for the knowledge base

Each folder here is one TRAIN scan you may build library entries from. None of these patients is ever read by the
tester. Each holds:

- `ct/` — the CT (ct.npz + meta.json), the same format the tester's scan has; open it with the program's
  `tools/view_ct.py` or render an entry with `tools/kb.py render`.
- `report.txt` — the radiologist's report the answers were read from.
- `answers.json` — the answer key for the five questions ("yes", "no", or "not scorable").
- `organs.json` — where each organ lies along the scan's long axis (TotalSegmentator, automatic).

`index.json` lists every scan with its answers, so you can find, say, all pneumonitis "yes" scans without opening
each folder.
"""


def answer_word(v) -> str:
    return "yes" if v == 1 else "no" if v == 0 else "not scorable"


def cmd_stage(args) -> int:
    dest = Path(args.dest)
    maps = {m["id"]: m for m in id_map() if m["role"] == "source"}
    source = json.loads((KB / "source.json").read_text())
    reports = KB / "source_reports"
    index, status = [], Counter()
    for s in source:
        rep = reports / f"{s['id']}.txt"
        if not rep.is_file():
            status["skipped_no_report"] += 1
            continue
        d = dest / s["id"]
        (d / "ct").mkdir(parents=True, exist_ok=True)
        for name in ("ct.npz", "meta.json"):
            if not (d / "ct" / name).is_file():
                tmp = d / "ct" / f"{name}.part"
                shutil.copy2(volume_dir(maps[s["id"]]) / name, tmp)
                tmp.replace(d / "ct" / name)
        answers = {k: answer_word(v) for k, v in s["answers"].items()}
        for name, body in (("report.txt", rep.read_text(encoding="utf-8")), ("answers.json", json.dumps(answers, indent=1)),
                           ("organs.json", (KB / "organs" / f"{s['id']}.json").read_text(encoding="utf-8"))):
            f = d / name
            if f.exists():
                f.chmod(0o644)
            f.write_text(body, encoding="utf-8")
        index.append({"id": s["id"], "answers": answers})
        status["staged"] += 1
    (dest / "index.json").write_text(json.dumps(index, indent=1))
    (dest / "README.md").write_text(SOURCE_README)
    for root, _dirs, files in os.walk(dest):
        for name in files:
            os.chmod(Path(root) / name, 0o444)
    print(f"source staged at {dest}: {dict(status)}")
    return 0


def cmd_check(_args) -> int:
    bad = []
    maps = id_map()
    roles = {r: {m["person_id"] for m in maps if m["role"] == r} for r in ("tester", "source")}
    if roles["tester"] & roles["source"]:
        bad.append("a patient is both tester and source")
    for r in roles:
        if len(roles[r]) != sum(m["role"] == r for m in maps):
            bad.append(f"{r}: two scans of one patient")
    val_patients = {m["person_id"] for m in map(json.loads, (bd.PRIVATE_DIR / "id_map.jsonl").read_text().splitlines())
                    if m and m["split"] == "val"}
    import pandas as pd
    test_patients = set(pd.read_parquet(bd.TEST_SAMPLE, columns=["person_id"])["person_id"].astype(int))
    if (roles["tester"] | roles["source"]) & (val_patients | test_patients):
        bad.append("a VAL or TEST-sample patient is in the KB sets")
    rounds = json.loads((KB / "rounds.json").read_text())
    tester_ids = {m["id"] for m in maps if m["role"] == "tester"}
    flat = [i for k, r in rounds.items() if not k.startswith("_") for i in r]
    if len(flat) != len(set(flat)) or set(flat) != tester_ids or len(flat) != TESTER_SIZE:
        bad.append("rounds do not split the tester set exactly once")
    gold = {json.loads(line)["id"]: json.loads(line) for line in (KB / "splits" / "kb_train.jsonl").read_text().splitlines() if line}
    per_round = []
    for k in sorted((k for k in rounds if not k.startswith("_")), key=int):
        c = Counter(klass(gold[i]) for i in rounds[k])
        per_round.append(dict(c))
        if len(rounds[k]) != PER_ROUND or c["P"] + c["N"] != PER_ROUND // 2:
            bad.append(f"round {k} has {len(rounds[k])} scans, {c['P'] + c['N']} positive")
    for m in maps:
        if not (volume_dir(m) / "ct.npz").is_file():
            bad.append(f"{m['role']} scan without a volume")
            break
    for m in maps:
        f = KB / "organs" / f"{m['id']}.json"
        if not f.is_file() or ".nii" in f.read_text() or m["id"] in f.read_text():
            bad.append("organ file missing or holds an identifier")
            break
    for i in tester_ids | set(val_scored_ids()):
        if not (KB / "similarity" / f"{i}.json").is_file():
            bad.append("a tester or VAL scan has no similarity file")
            break
    src = [m for m in maps if m["role"] == "source"]
    n_rep = sum((KB / "source_reports" / f"{m['id']}.txt").is_file() for m in src)
    print(f"tester {len(tester_ids)} scans / {len(roles['tester'])} patients; source {len(src)} / {len(roles['source'])}; "
          f"source with a report {n_rep}; per round {per_round}")
    if n_rep < 0.9 * len(src):
        bad.append(f"only {n_rep} of {len(src)} source scans have a report")
    if bad:
        print(f"FAIL: {bad[:5]}")
        return 1
    print("PASS: KB data ready")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    v = sub.add_parser("volumes")
    v.add_argument("--workers", type=int, default=3)
    for name in ("reports", "organs", "similarity", "check"):
        sub.add_parser(name)
    s = sub.add_parser("stage")
    s.add_argument("dest")
    args = ap.parse_args()
    return {"select": cmd_select, "volumes": cmd_volumes, "reports": cmd_reports, "organs": cmd_organs,
            "similarity": cmd_similarity, "stage": cmd_stage, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
