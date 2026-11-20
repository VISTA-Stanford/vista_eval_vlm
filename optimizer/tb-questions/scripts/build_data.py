"""Build the tb-questions task data for one agent (TB_AGENT=treatment or imaging). Prints counts only: never an id, a
UID, a date, a row or any summary text.

Treatment (spec planning/vista_eval_vlm/2026-11-20-tb-agent-questions.md, Decided): vista_bench v1.7
``tb_classification_tasks``, tasks ``early_stage_management_answer`` and ``optimal_systemic_therapy_answer``, rows with
a scored answer (label 0-4; -1 Indeterminate/Deferred is not scored) and a tumor board date; the official split (train
-> TRAIN, valid -> VAL, test sealed). One row per patient with the answer code for each question the tumor board
answered (null for the other) and the tumor board-time ``compact_summary`` (cut one week before the first tumor
board) as patients/<id>/summary.txt. Stops if a patient has two different summaries or answers, or no summary.

Imaging: the eval set's step-4 list (vistabench/vlm/evalset_v1/samples/step4.parquet: the 334 tumor board cases with a
scored progression answer, the most recent CT before the tumor board and its prior CT from per_ct_prior_ct_link, the
pairs every VLM answered). Its 288 train + valid pairs are pooled and redrawn into VAL_SIZE picking (VAL) and the rest
feedback (TRAIN), stratified by the tumor board's answer, with a fixed seed (Phil 2026-11-20). TEST = the 45 in
step4_test.parquet, untouched. The one case with no split is left out. Both CTs of each pair must be on the mount.

    select   write splits/{train,val}.jsonl, pools.json, the private id map and (treatment) the summaries
    volumes  (imaging) convert each pair's NIfTIs into patients/<id>/{prior,current}/ct.npz + meta.json, exactly as
             ct-findings-5 does (HU int16, RAS, thin scans averaged into ~2.5 mm slabs, no header, date or UID kept)
    test     write the TEST split (treatment: the 119 TEST patients of step6_test.parquet's list, same rule; imaging:
             the 45 of step4_test.parquet), with its own private id map; imaging then needs ``volumes --split test``.
             Only on Phil's go (spec: the TEST look needs its own go).

    cd optimizer/tb-questions
    TB_AGENT=treatment uv run -q --no-project --with google-cloud-bigquery --with pandas --with pyarrow --with db-dtypes python -B scripts/build_data.py select
    TB_AGENT=imaging uv run -q --no-project --with pandas --with pyarrow python -B scripts/build_data.py select
    TB_AGENT=imaging OMP_NUM_THREADS=1 nice -n 19 uv run -q --no-project --with nibabel --with numpy python -B scripts/build_data.py volumes --workers 4

BigQuery runs on this host's application-default login. The summaries and CTs are written to the shared mount (inside
the secure environment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

AGENT = os.environ.get("TB_AGENT", "")
if AGENT not in ("treatment", "imaging"):
    raise SystemExit("set TB_AGENT=treatment or TB_AGENT=imaging")
PROJECT = "som-nero-plevriti-deidbdf"
TABLE = f"{PROJECT}.vista_bench_v1_7.tb_classification_tasks"
#: (answer key, vista_bench task), in config.QUESTIONS order.
TREATMENT_QUESTIONS = (("early_stage_plan", "early_stage_management_answer"),
                       ("first_line_regimen", "optimal_systemic_therapy_answer"))
PROGRESSION = ("progression", "progression_assessment_answer")
SAMPLES = Path("/mnt/su-vista-uscentral1/vistabench/vlm/evalset_v1/samples")
NIFTI_ROOT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/ct_data/ct_scans/vista/feb26")  # context.nifti's snapshot
MOUNT = Path("/mnt/su-vista-uscentral1/chaudhari_lab/phil")
DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", MOUNT / "tb-questions-v1_7" / AGENT))
PRIVATE_DIR = Path(os.environ.get("TASK_PRIVATE_DIR", MOUNT / "tb-questions-v1_7-private" / AGENT))
SALT = f"tb-questions-v1:{AGENT}"
#: Imaging: the picking pool's size out of the 288 (the rest is feedback), and the draw's seed (spec: ~188 / ~100).
VAL_SIZE = 100
SEED = "tb-questions-v1:imaging:redraw"
SLAB_MM = 2.5


def opaque_id(person_id) -> str:
    return hashlib.sha1(f"{SALT}:{int(person_id)}".encode()).hexdigest()[:16]


def nifti(study: str, series: str) -> Path:
    return NIFTI_ROOT / f"{study}__{series}.nii.gz"


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")


def answer_counts(rows: list[dict], keys) -> dict:
    return {k: dict(sorted(Counter(str(r[k]) for r in rows if r[k] is not None).items())) for k in keys}


# ==================================================================================================
# Treatment
# ==================================================================================================


def treatment_rows(splits: tuple[str, ...]) -> dict[str, list[dict]]:
    """{table split: [{person_id, labels: {key: code | None}, summary}]} for the given table splits."""
    from google.cloud import bigquery

    sql = f"""select person_id, split, task, label, compact_summary
        from `{TABLE}`
        where task in unnest(@tasks) and split in unnest(@splits) and embed_time is not null and label >= 0"""
    job = bigquery.Client(project=PROJECT).query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("tasks", "STRING", [t for _k, t in TREATMENT_QUESTIONS]),
        bigquery.ArrayQueryParameter("splits", "STRING", list(splits))]))
    key_of = {t: k for k, t in TREATMENT_QUESTIONS}
    people: dict[int, dict] = {}
    for r in job.result():
        p = people.setdefault(int(r["person_id"]), {"person_id": int(r["person_id"]), "split": r["split"],
                                                    "labels": dict.fromkeys(key_of.values()), "summary": None})
        if p["split"] != r["split"]:
            raise SystemExit("one patient in two splits: the table's contract changed")
        k = key_of[r["task"]]
        if p["labels"][k] is not None and p["labels"][k] != int(r["label"]):
            raise SystemExit("one patient with two different answers to one question: the table's contract changed")
        p["labels"][k] = int(r["label"])
        text = (r["compact_summary"] or "").strip()
        if not text:
            raise SystemExit("a scored tumor board row without a tumor board-time summary")
        if p["summary"] is not None and p["summary"] != text:
            raise SystemExit("a patient with different summaries on different tumor board rows")
        p["summary"] = text
    out: dict[str, list[dict]] = {s: [] for s in splits}
    for p in people.values():
        out[p["split"]].append(p)
    return out


def write_treatment(split: str, people: list[dict], id_map: list[dict]) -> list[dict]:
    rows = []
    for p in people:
        sid = opaque_id(p["person_id"])
        d = DATA_DIR / "patients" / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.txt").write_text(p["summary"] + "\n", encoding="utf-8")
        rows.append({"id": sid, **p["labels"]})
        id_map.append({"id": sid, "person_id": p["person_id"], "split": split})
    rows.sort(key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("two patients share an opaque id")
    return rows


def treatment_select() -> int:
    import pandas as pd

    by = treatment_rows(("train", "valid"))
    test_people = set(pd.read_parquet(SAMPLES / "step6_test.parquet", columns=["person_id"])["person_id"].astype(int))
    tv = {p["person_id"] for ps in by.values() for p in ps}
    if {p["person_id"] for p in by["train"]} & {p["person_id"] for p in by["valid"]}:
        raise SystemExit("a patient in both train and valid")
    if tv & test_people:
        raise SystemExit(f"STOP: {len(tv & test_people)} train/valid patients are in the tumor board TEST list")
    id_map: list[dict] = []
    rows = {"train": write_treatment("train", by["train"], id_map), "val": write_treatment("val", by["valid"], id_map)}
    keys = [k for k, _t in TREATMENT_QUESTIONS]
    for split, items in rows.items():
        write_jsonl(DATA_DIR / "splits" / f"{split}.jsonl", items)
    write_jsonl(PRIVATE_DIR / "id_map.jsonl", id_map)
    counts = {s: {"patients": len(items), "both_questions": sum(all(r[k] is not None for k in keys) for r in items),
                  "answers": answer_counts(items, keys)} for s, items in rows.items()}
    (DATA_DIR / "pools.json").write_text(json.dumps({
        **{s: [r["id"] for r in items] for s, items in rows.items()},
        "_draw": {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "table": TABLE,
                  "rule": "official split: train -> TRAIN (feedback), valid -> VAL (picking); every scored patient",
                  "salt": SALT, "test_list_patients": len(test_people), "test_overlap": 0, "counts": counts}},
        indent=1), encoding="utf-8")
    print(f"tumor board TEST list: {len(test_people)} patients, 0 in train/valid")
    for s, c in counts.items():
        print(f"{s}: {c['patients']} patients ({c['both_questions']} asked both questions); answers by code {c['answers']}")
    return 0


def treatment_test() -> int:
    import pandas as pd

    by = treatment_rows(("test",))
    want = set(pd.read_parquet(SAMPLES / "step6_test.parquet", columns=["person_id"])["person_id"].astype(int))
    got = {p["person_id"] for p in by["test"]}
    if got != want:
        raise SystemExit(f"STOP: TEST patients differ from step6_test.parquet ({len(got)} vs {len(want)})")
    pool = {int(m["person_id"]) for m in map(json.loads, (PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    if pool & got:
        raise SystemExit("STOP: a TEST patient is in the TRAIN/VAL pools")
    id_map: list[dict] = []
    rows = write_treatment("test", by["test"], id_map)
    write_jsonl(PRIVATE_DIR / "id_map_test.jsonl", id_map)
    write_jsonl(DATA_DIR / "splits" / "test.jsonl", rows)
    print(f"test: {len(rows)} patients (== step6_test.parquet); answers by code "
          f"{answer_counts(rows, [k for k, _t in TREATMENT_QUESTIONS])}")
    return 0


# ==================================================================================================
# Imaging
# ==================================================================================================


def step4(name: str):
    import pandas as pd

    d = pd.read_parquet(SAMPLES / f"{name}.parquet")
    lab = f"label__{PROGRESSION[1]}"
    if d["person_id"].duplicated().any() or not d[lab].isin([0, 1]).all():
        raise SystemExit(f"{name}: a repeated patient or an unscored answer: the list's contract changed")
    return d, lab


def pair_entry(r, split: str) -> dict:
    cur, pri = nifti(r.study, r.series), nifti(r.prior_study, r.prior_series)
    if not (cur.is_file() and pri.is_file()):
        raise SystemExit("STOP: a pair whose CT or prior CT is not on the mount")
    return {"id": opaque_id(r.person_id), "person_id": int(r.person_id), "split": split, "table_split": r.split,
            "current": str(cur), "prior": str(pri)}


def imaging_select() -> int:
    d, lab = step4("step4")
    test = set(step4("step4_test")[0]["person_id"].astype(int))
    pool = d[d["split"].isin(["train", "valid"])]
    if set(pool["person_id"].astype(int)) & test:
        raise SystemExit("STOP: a train/valid pair's patient is in the 45 TEST pairs")
    # Stratified by answer: each answer gives VAL its share of VAL_SIZE (largest remainder), from a seeded shuffle of
    # its patients in opaque-id order.
    rng = random.Random(SEED)
    groups = {y: sorted(pool[pool[lab] == y].itertuples(index=False), key=lambda r: opaque_id(r.person_id))
              for y in (0, 1)}
    exact = {y: VAL_SIZE * len(g) / len(pool) for y, g in groups.items()}
    take = {y: int(v) for y, v in exact.items()}
    for y in sorted(exact, key=lambda y: exact[y] - take[y], reverse=True)[:VAL_SIZE - sum(take.values())]:
        take[y] += 1
    rows, id_map = {"train": [], "val": []}, []
    for y, g in groups.items():
        rng.shuffle(g)
        for n, r in enumerate(g):
            split = "val" if n < take[y] else "train"
            e = pair_entry(r, split)
            rows[split].append({"id": e["id"], PROGRESSION[0]: int(getattr(r, lab))})
            id_map.append(e)
    for split, items in rows.items():
        items.sort(key=lambda r: r["id"])
        write_jsonl(DATA_DIR / "splits" / f"{split}.jsonl", items)
    if len({e["id"] for e in id_map}) != len(id_map):
        raise SystemExit("two patients share an opaque id")
    write_jsonl(PRIVATE_DIR / "id_map.jsonl", id_map)
    counts = {s: {"pairs": len(items), "yes": sum(r[PROGRESSION[0]] for r in items),
                  "from_table_split": dict(Counter(e["table_split"] for e in id_map if e["split"] == s))}
              for s, items in rows.items()}
    (DATA_DIR / "pools.json").write_text(json.dumps({
        **{s: [r["id"] for r in items] for s, items in rows.items()},
        "_draw": {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "source": str(SAMPLES / "step4.parquet"), "seed": SEED, "val_size": VAL_SIZE, "salt": SALT,
                  "rule": ("the step-4 list's train + valid pairs pooled and redrawn: per tumor board answer, a seeded "
                           "shuffle in opaque-id order gives VAL its share of VAL_SIZE (largest remainder), the rest "
                           "TRAIN"), "left_out_no_split": int((d["split"].fillna("") == "").sum()),
                  "test_pairs": len(test), "test_overlap": 0, "counts": counts}}, indent=1), encoding="utf-8")
    print(f"step-4 list: {len(d)} pairs; pooled train + valid {len(pool)}; TEST {len(test)} (0 overlap); "
          f"no split {int((d['split'].fillna('') == '').sum())} (left out)")
    for s, c in counts.items():
        print(f"{s}: {c['pairs']} pairs, {c['yes']} progression ({c['yes'] / c['pairs']:.0%}); from {c['from_table_split']}")
    return 0


def imaging_test() -> int:
    d, lab = step4("step4_test")
    pool = {int(m["person_id"]) for m in map(json.loads, (PRIVATE_DIR / "id_map.jsonl").read_text().splitlines()) if m}
    if pool & set(d["person_id"].astype(int)):
        raise SystemExit("STOP: a TEST patient is in the TRAIN/VAL pools")
    id_map = [pair_entry(r, "test") for r in d.itertuples(index=False)]
    rows = sorted(({"id": e["id"], PROGRESSION[0]: int(y)} for e, y in zip(id_map, d[lab])), key=lambda r: r["id"])
    write_jsonl(PRIVATE_DIR / "id_map_test.jsonl", id_map)
    write_jsonl(DATA_DIR / "splits" / "test.jsonl", rows)
    print(f"test: {len(rows)} pairs, {sum(r[PROGRESSION[0]] for r in rows)} progression; next: volumes --split test")
    return 0


def convert(item: tuple[str, str]) -> tuple[str, str]:
    """One NIfTI -> <out>/ct.npz + meta.json (ct-findings-5's converter). Returns (status, detail), no ids."""
    import nibabel as nib
    import numpy as np

    out_dir, src = item
    out = Path(out_dir)
    if (out / "meta.json").is_file():
        return "kept", ""
    try:
        img = nib.as_closest_canonical(nib.load(src))
        hu = np.asarray(img.dataobj, dtype=np.float32)
        if hu.ndim == 4:
            hu = hu[..., 0]
        sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
        k = max(1, int(round(SLAB_MM / sz))) if sz < SLAB_MM else 1
        if k > 1:
            n = hu.shape[2] // k * k
            hu = hu[:, :, :n].reshape(hu.shape[0], hu.shape[1], n // k, k).mean(axis=3)
            sz *= k
        hu = np.clip(np.rint(hu), -1024, 3071).astype(np.int16)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "ct.tmp.npz"
        np.savez_compressed(tmp, hu=hu)
        tmp.replace(out / "ct.npz")
        meta = {
            "array": "ct.npz, key 'hu': Hounsfield units, int16, shape (x, y, z)",
            "shape": list(hu.shape),
            "spacing_mm": [round(sx, 4), round(sy, 4), round(sz, 4)],
            "orientation": ("RAS: index increases toward the patient's Right along axis 0 (x), Anterior along "
                            "axis 1 (y), Superior along axis 2 (z); z = 0 is the lowest (most inferior) slice"),
            "slab_averaged": k,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return "converted", ""
    except Exception as exc:  # noqa: BLE001 - one bad volume must not stop the batch; reported as a count
        return "failed", type(exc).__name__


def cmd_volumes(args) -> int:
    if AGENT != "imaging":
        raise SystemExit("volumes is for TB_AGENT=imaging")
    name = "id_map_test.jsonl" if args.split == "test" else "id_map.jsonl"
    entries = [m for m in map(json.loads, (PRIVATE_DIR / name).read_text().splitlines()) if m]
    if args.limit:
        entries = entries[: args.limit]
    items = [(str(DATA_DIR / "patients" / e["id"] / scan), e[scan]) for e in entries for scan in ("prior", "current")]
    status, errors = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (st, detail) in enumerate(pool.map(convert, items), 1):
            status[st] += 1
            if detail:
                errors[detail] += 1
            if n % 50 == 0:
                print(f"{n}/{len(items)} {dict(status)}", flush=True)
    print(f"done {len(items)} volumes ({len(entries)} pairs): {dict(status)}; errors by type {dict(errors)}")
    return 1 if status["failed"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    v = sub.add_parser("volumes")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--split", choices=("pools", "test"), default="pools")
    sub.add_parser("test")
    args = ap.parse_args()
    if args.cmd == "volumes":
        return cmd_volumes(args)
    return {("select", "treatment"): treatment_select, ("select", "imaging"): imaging_select,
            ("test", "treatment"): treatment_test, ("test", "imaging"): imaging_test}[(args.cmd, AGENT)]()


if __name__ == "__main__":
    sys.exit(main())
