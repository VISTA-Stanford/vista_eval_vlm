"""Pull Tuomo's patient summaries for the summary input arm (config.WITH_SUMMARY; Phil 2026-10-29).

One summary per patient, from llm_labels.text_llm_summary, taking only the row dated exactly at this task's cutoff
(first treatment - 7 days, PRIVATE/cohort.parquet): a patient can also have rows made for other tasks at later
dates, and those would leak. Among same-day rows the prognosis one (task_groups holds
progression_recurrence_survival) wins. The text is rich_summary when has_rich, else compact_summary.

Patients: the TRAIN pool (400), and every VAL patient (811, which holds the 200-patient VAL pool). Never TEST.
Writes MOUNT_DATA_DIR/summaries/<id>.txt (the durable copy, inside the secure environment) and the same files into
config.SUMMARY_DIR (the local working copy), read-only. A PHI write: Phil's go 2026-10-29.

    uv run --with 'duckdb<2' --with google-cloud-bigquery python scripts/pull_summaries.py          (BigQuery)
    uv run --with 'duckdb<2' python scripts/pull_summaries.py --local-only   (copy the mount's files to SUMMARY_DIR)

Prints counts only: never an id, a date or any text.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TABLE = "som-nero-plevriti-deidbdf.llm_labels.text_llm_summary"
PROGNOSIS = "progression_recurrence_survival"
MOUNT_DIR = config.MOUNT_DATA_DIR / "summaries"


def wanted_ids() -> list[str]:
    pools = json.loads((config.DATA_DIR / "pools.json").read_text(encoding="utf-8"))
    val = [json.loads(line)["id"] for line in (config.DATA_DIR / "splits" / "val.jsonl").read_text().splitlines() if line]
    return sorted(set(pools["train"]) | set(pools["val"]) | set(val))


def write_ro(path: Path, text: str) -> None:
    if path.exists():
        path.chmod(0o644)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o444)


def pull(ids: list[str]) -> None:
    import duckdb
    from google.cloud import bigquery

    cohort = {i: (p, c) for i, p, c in duckdb.sql(
        f"select id, person_id, cutoff from '{config.PRIVATE_DIR / 'cohort.parquet'}'").fetchall()}
    missing_map = [i for i in ids if i not in cohort]
    if missing_map:
        raise SystemExit(f"{len(missing_map)} ids have no row in cohort.parquet")
    pids = sorted({cohort[i][0] for i in ids})
    rows = list(bigquery.Client(project="som-nero-plevriti-deidbdf").query(
        f"select person_id, cast(embed_time as date) d, cast(cutoff_inclusive as string) ci, task_groups, has_rich, "
        f"rich_summary, compact_summary from `{TABLE}` where person_id in unnest(@p)",
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("p", "INT64", pids)])).result())
    per: dict[int, list] = {}
    for r in rows:
        per.setdefault(r.person_id, []).append(r)

    MOUNT_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for i in ids:
        pid, cut = cohort[i]
        same = [r for r in per.get(pid, []) if r.d == cut]
        counts["later-dated rows skipped"] += sum(1 for r in per.get(pid, []) if r.d > cut)
        if not same:
            counts["no summary at the cutoff"] += 1
            continue
        same.sort(key=lambda r: PROGNOSIS not in str(r.task_groups))
        r = same[0]
        counts[f"cutoff_inclusive={r.ci}"] += 1
        text = r.rich_summary if (r.has_rich and r.rich_summary) else r.compact_summary
        if not text:
            counts["empty text"] += 1
            continue
        counts["rich" if (r.has_rich and r.rich_summary) else "compact"] += 1
        counts["prognosis row" if PROGNOSIS in str(r.task_groups) else "other task row, same day"] += 1
        write_ro(MOUNT_DIR / f"{i}.txt", text)
        counts["written"] += 1
    print(f"patients asked: {len(ids)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


def copy_local(ids: list[str]) -> None:
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in ids:
        src = MOUNT_DIR / f"{i}.txt"
        if src.is_file():
            write_ro(config.SUMMARY_DIR / f"{i}.txt", src.read_text(encoding="utf-8"))
            n += 1
    print(f"local copies in SUMMARY_DIR: {n} of {len(ids)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local-only", action="store_true", help="no BigQuery: copy the mount's summaries to SUMMARY_DIR")
    args = p.parse_args()
    ids = wanted_ids()
    if not args.local_only:
        pull(ids)
    copy_local(ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
