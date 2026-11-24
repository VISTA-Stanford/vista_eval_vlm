"""The retrieval arm's readback numbers (spec planning/vista_eval_vlm/2026-11-17-ct-findings-retrieval-arm.md): how the
sessions used their neighbour folders, and the retrieval pass against every untuned VAL-200 pass, scan by scan.

    usage    from one pass's transcripts (beside its answers.json, as the ProgramRunner keeps them; run it on the
             machine that ran the pass): per scan, whether the session touched neighbours/ at all, how many times it
             rendered a neighbour's CT (view_ct.py on neighbours/n<k>/), how many distinct neighbours it viewed, and
             how many times it read a neighbour report (report.txt) or the answers (answers.json, neighbours.json,
             README.md). Writes per-scan counts (opaque ids, no text) and a summary; flags a pass where most scans
             never touched the folder (the spec's stop-and-come-back).
    compare  the retrieval pass against each untuned pass (answers.json files): per question and mean AUROC of each
             (the task's own scorer: a failed answer counts 0.5); the per-scan paired bootstrap 95% interval of
             retrieval minus each untuned pass (scans resampled together, 2000 draws); the untuned passes' spread
             (range of their means, and each pair's paired interval) as the noise bar, and the verdict: the arm
             differs only if every paired interval excludes 0 and every difference beats the largest untuned pair; and, per question, how often
             each pass's answer (a stated chance of 50 or more = yes) agrees with the neighbours' majority (ties and no answers left out),
             with the mean gap between the stated probability and the neighbours' yes share. With --baselines (the
             JSON of score_baselines.py score --out), the probe and vote rows join the table. Writes
             compare.json and table.md to --out.
    arms     the example-CT arms (spec 2026-11-22): each --pass scored on the VAL pool scans it answers, all passes on
             the scans every pass answers, and each --pair X:Y as X minus Y paired over the scans both answer. Writes
             arms.json and arms.md to --out.

    cd optimizer/ct-findings-5
    uv run -q --project $AGENTIC_LABEL_OPT python -B scripts/retrieval_usage.py usage ANSWERS_JSON --out DIR
    uv run -q --project $AGENTIC_LABEL_OPT python -B scripts/retrieval_usage.py compare --retrieval ANSWERS_JSON \
        --untuned prob-r1=A.json --untuned prob-r2=B.json --untuned prob-r4=C.json --baselines BASE.json --out DIR
    uv run -q --project $AGENTIC_LABEL_OPT python -B scripts/retrieval_usage.py arms --pass A=a.json --pass B=b.json \
        --pair A:B --out DIR

Counts and AUROCs only: never a report line, a finding note or an identifier other than the opaque scan id.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import config  # noqa: E402  (also puts the engine on sys.path)
import adapters as ad  # noqa: E402  (the scorer's own AUROC and failed-answer rule)

KEYS = tuple(k for k, _t, _q in config.QUESTIONS)
#: The organ-zoom arms' deciding mean (spec 2026-11-24): the three questions where small findings matter.
SMALL_KEYS = ("lesion_liver", "lesion_lungs", "lesion_lymph_nodes")
BOOTSTRAP = 2000
SEED = 0
NEIGHBOUR_PATH = re.compile(r"neighbours/n(\d)")
WORD = {"yes": 1, "no": 0}


# --------------------------------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------------------------------


def events_file(pass_dir: Path, call_id: str) -> Path | None:
    """The latest attempt's events.jsonl for one call (adapters._trace_files' layout)."""
    root = pass_dir / "transcripts" / call_id
    attempts = sorted(root.glob("attempt*"), key=lambda p: int(p.name.removeprefix("attempt") or 0))
    runs = sorted(attempts[-1].glob("*/*")) if attempts else []
    f = runs[-1] / "events.jsonl" if runs else None
    return f if f and f.is_file() else None


def tool_inputs(events: Path):
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "assistant":
            continue
        for c in e.get("message", {}).get("content", []):
            if isinstance(c, dict) and c.get("type") == "tool_use":
                yield c.get("name"), json.dumps(c.get("input", {}))


#: Only these tools read; Write and Edit are the session's own notes and answers, never counted as using the folder.
READING_TOOLS = ("Read", "Bash", "Grep", "Glob")
#: A render or load of a CT: the viewer (not --help or --info, which draw nothing) or Python loading a ct.npz.
VIEWER = re.compile(r"view_ct(?:\.py)?\b")
PIECES = re.compile(r"&&|\|\||;|\n|\|")
#: Arm D's pre-rendered neighbour grids (scripts/build_views.py), opened with the Read tool.
GRID_READ = re.compile(r"neighbours/n([1-5])/views/(\w+)\.png")
#: Neighbour ranks a shell loop or glob covers: "for k in 1 2 3", "n{1..5}", "n*".
LOOP_LIST = re.compile(r"for\s+\w+\s+in\s+([1-5](?:\s+[1-5])*)\b")


def _loop_ranks(command: str, piece: str) -> list[str]:
    m = LOOP_LIST.search(command)
    if m:
        return m.group(1).split()
    if re.search(r"n\{1\.\.5\}|neighbours/n\*|(?<![\w/])n\*", piece):
        return ["1", "2", "3", "4", "5"]
    return []


def neighbour_views(command: str) -> list[str]:
    """The neighbour ranks one Bash command renders or loads, one entry per view. Fixes the 11-17 counter, which
    counted any command naming both view_ct and neighbours (e.g. `view_ct.py --help; cat neighbours/README.md`):
    each shell piece is judged alone, a `cd` into neighbours/ or neighbours/n<k> is followed, and --help, --info and
    the session's own ct/ never count. A loop or glob over the neighbours counts each rank it covers once."""
    views, cwd = [], None  # cwd: None outside neighbours/, "0" in neighbours/, "<k>" in neighbours/n<k>
    for piece in (p.strip() for p in PIECES.split(command)):
        cd = re.match(r"cd\s+(\S+)", piece)
        if cd:
            t = cd.group(1).rstrip("/")
            k = re.search(r"neighbours/n([1-5])$", t)
            if k:
                cwd = k.group(1)
            elif t.endswith("neighbours"):
                cwd = "0"
            elif cwd == "0" and re.fullmatch(r"n[1-5]", t):
                cwd = t[1]
            else:
                cwd = None
            continue
        viewer = VIEWER.search(piece) and "--help" not in piece and "--info" not in piece
        loader = "ct.npz" in piece and ("load" in piece or "python" in piece)
        if not (viewer or loader):
            continue
        ranks = re.findall(r"neighbours/n([1-5])\b", piece)
        if not ranks and cwd == "0":
            ranks = re.findall(r"(?<![\w/])n([1-5])/", piece)
        if not ranks and cwd not in (None, "0") and re.search(r"(?<![\w/])(?:\./)?ct\b", piece):
            ranks = [cwd]
        if not ranks and ("neighbours/" in piece or cwd == "0"):
            ranks = _loop_ranks(command, piece)
        views += ranks
    return views


def count_one(events: Path) -> dict:
    """One session's use of neighbours/, from its reading tool calls. A view: a rank neighbour_views finds in a Bash
    call. A report read: a call naming report.txt, or any grep (Bash or the Grep tool) over neighbours/ (the reports
    are the folder's only free text). An answers read: answers.json, neighbours.json or README.md under
    neighbours/."""
    out = {"touched": False, "neighbour_views": 0, "neighbours_viewed": 0, "report_reads": 0, "answer_reads": 0,
           "tool_calls": 0, "neighbour_grids_opened": 0}
    viewed, grids = set(), set()
    for name, text in tool_inputs(events):
        out["tool_calls"] += 1
        if name not in READING_TOOLS or "neighbours" not in text and not (name == "Bash" and "cd n" in text):
            continue
        out["touched"] = True
        if name == "Read":
            g = GRID_READ.search(json.loads(text).get("file_path", ""))
            if g:  # a view of that neighbour, like a render; distinct (rank, grid) pairs for the all-15 count
                out["neighbour_views"] += 1
                viewed.add(g.group(1))
                grids.add(g.groups())
        if name == "Bash":
            cmd = json.loads(text).get("command", "")
            v = neighbour_views(cmd)
            out["neighbour_views"] += len(v)
            viewed.update(v)
        if "report.txt" in text or name == "Grep" or (name == "Bash" and "grep" in text):
            out["report_reads"] += 1
        if any(n in text for n in ("answers.json", "neighbours.json", "README.md")):
            out["answer_reads"] += 1
    out["neighbours_viewed"] = len(viewed)
    out["neighbour_grids_opened"] = len(grids)
    return out


def cmd_usage(args) -> int:
    answers_path = Path(args.answers)
    answers = json.loads(answers_path.read_text(encoding="utf-8"))["answers"]
    per, missing = {}, 0
    for cid in sorted(answers):
        f = events_file(answers_path.parent, cid)
        if f is None:
            missing += 1
            continue
        per[cid] = count_one(f)
    n = len(per)
    if not n:
        print(f"FAIL: no transcripts beside {answers_path}")
        return 1

    def share(key):
        return sum(1 for v in per.values() if v[key]) / n

    summary = {
        "scans": len(answers), "with_transcript": n, "without_transcript": missing,
        "share_touched_folder": round(share("touched"), 3),
        "share_viewed_a_neighbour_ct": round(share("neighbour_views"), 3),
        "share_read_a_report": round(share("report_reads"), 3),
        "share_read_answers": round(share("answer_reads"), 3),
        "share_opened_all_15_grids": round(sum(1 for v in per.values() if v["neighbour_grids_opened"] >= 15) / n, 3),
        **{f"median_{k}": statistics.median(v[k] for v in per.values())
           for k in ("neighbour_views", "neighbours_viewed", "report_reads", "answer_reads", "tool_calls",
                     "neighbour_grids_opened")},
        **{f"mean_{k}": round(statistics.mean(v[k] for v in per.values()), 2)
           for k in ("neighbour_views", "neighbours_viewed", "report_reads", "answer_reads", "tool_calls",
                     "neighbour_grids_opened")},
    }
    summary["ignored_on_most_scans"] = summary["share_touched_folder"] < 0.5
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "usage_per_scan.json").write_text(json.dumps(per, indent=0), encoding="utf-8")
    (out / "usage_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for k, v in summary.items():
        print(f"  {k:<32} {v}")
    if summary["ignored_on_most_scans"]:
        print("STOP AND COME BACK: most sessions never touched the neighbour folder (spec)")
    print(f"written usage_per_scan.json and usage_summary.json to {out}")
    return 0


# --------------------------------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------------------------------


def load_answers(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["answers"]


def aurocs(gold: dict, answers: dict, ids: list[str]) -> dict:
    """Per question AUROC over ``ids`` (the scorer's rule: scored scans only, a failed answer 0.5), and the mean."""
    out = {}
    for k in KEYS:
        pairs = [(gold[i][k], ad.prob(answers[i], k)) for i in ids if gold[i][k] is not None]
        out[k] = ad.auroc(pairs)
    out["mean"] = sum(out[k] for k in KEYS) / len(KEYS)
    out["mean3"] = sum(out[k] for k in SMALL_KEYS) / len(SMALL_KEYS)
    return out


def paired(gold: dict, a: dict, b: dict, ids: list[str]) -> dict:
    """a minus b: the point difference and the 95% percentile interval over scan resamples (one draw of scans feeds
    both passes), per question and for the mean."""
    rng = random.Random(SEED)
    point_a, point_b = aurocs(gold, a, ids), aurocs(gold, b, ids)
    draws = {k: [] for k in (*KEYS, "mean", "mean3")}
    for _ in range(BOOTSTRAP):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        sa, sb = aurocs(gold, a, sample), aurocs(gold, b, sample)
        for k in draws:
            draws[k].append(sa[k] - sb[k])
    out = {}
    for k, vals in draws.items():
        vals.sort()
        out[k] = {"diff": round(point_a[k] - point_b[k], 4), "lower": round(vals[int(0.025 * BOOTSTRAP)], 4),
                  "upper": round(vals[int(0.975 * BOOTSTRAP) - 1], 4)}
    return out


def neighbour_shares(ids: list[str]) -> dict:
    """Per scan and question, the neighbours' yes share over those with an answer (None when none has one)."""
    out = {}
    for i in ids:
        said = [json.loads((config.NEIGHBOURS_DIR / i / f"n{r}" / "answers.json").read_text()) for r in range(1, 6)]
        out[i] = {}
        for k in KEYS:
            ys = [WORD[s[k]] for s in said if s[k] in WORD]
            out[i][k] = sum(ys) / len(ys) if ys else None
    return out


def agreement(answers: dict, shares: dict, ids: list[str]) -> dict:
    out = {}
    for k in KEYS:
        agree = n = 0
        gaps = []
        for i in ids:
            s = shares[i][k]
            if s is None or answers[i].get("error"):
                continue
            p = ad.prob(answers[i], k)
            gaps.append(abs(p - s))
            if s == 0.5:
                continue
            n += 1
            agree += (p >= 0.5) == (s > 0.5)
        out[k] = {"with_majority": n, "agree_share": round(agree / n, 3) if n else None,
                  "mean_gap_to_neighbour_share": round(statistics.mean(gaps), 3) if gaps else None}
    return out


def cmd_compare(args) -> int:
    gold = {r["id"]: r for r in ad.read_split("val")}
    ids = sorted(ad.pools()["val"])
    retrieval = load_answers(args.retrieval)
    untuned = {}
    for item in args.untuned:
        name, _, path = item.partition("=")
        untuned[name] = load_answers(path)
    for name, ans in [("retrieval", retrieval), *untuned.items()]:
        if set(ans) != set(ids):
            raise SystemExit(f"{name}: answers cover {len(set(ans) & set(ids))} of the {len(ids)} VAL pool scans")
    passes = {"retrieval": retrieval, **untuned}
    scores = {n: aurocs(gold, a, ids) for n, a in passes.items()}
    errors = {n: sum(1 for i in ids if a[i].get("error")) for n, a in passes.items()}
    vs_untuned = {n: paired(gold, retrieval, a, ids) for n, a in untuned.items()}
    names = list(untuned)
    noise = {f"{x} - {y}": paired(gold, untuned[x], untuned[y], ids) for xi, x in enumerate(names) for y in names[xi + 1:]}
    means = [scores[n]["mean"] for n in names]
    shares = neighbour_shares(ids)
    agree = {n: agreement(a, shares, ids) for n, a in passes.items()}
    baselines = {}
    if args.baselines:
        base = json.loads(Path(args.baselines).read_text())
        for row in base["baselines"]:
            baselines.setdefault(row["baseline"], {})[row["question"]] = row.get("auroc")
        for name, qs in baselines.items():
            qs["mean"] = base["mean_auroc"].get(name)
    band = (min(means), max(means)) if means else None
    # The noise verdict (spec: "lands inside the untuned noise band" is a result to report): the retrieval arm
    # differs from the untuned agent only when every paired interval against an untuned pass excludes 0 and every
    # difference is larger than the largest difference between two untuned passes. Otherwise it is inside the noise.
    widest = max((abs(d["mean"]["diff"]) for d in noise.values()), default=0.0)
    beyond = [d["mean"] for d in vs_untuned.values()]
    differs = bool(beyond) and all((d["lower"] > 0 or d["upper"] < 0) and abs(d["diff"]) > widest for d in beyond)
    inside = not differs
    result = {"n": len(ids), "scores": scores, "program_errors": errors, "retrieval_minus_untuned": vs_untuned,
              "untuned_pairs": noise, "untuned_mean_range": band, "largest_untuned_pair_diff": widest, "retrieval_inside_untuned_noise": inside,
              "agreement_with_neighbour_majority": agree, "baselines": baselines, "bootstrap": BOOTSTRAP}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    cols = (*KEYS, "mean")
    head = "| row | " + " | ".join(c.replace("lesion_", "") for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for n in names:
        lines.append(f"| untuned {n} | " + " | ".join(f"{scores[n][c]:.3f}" for c in cols) + " |")
    lines.append("| **retrieval arm** | " + " | ".join(f"**{scores['retrieval'][c]:.3f}**" for c in cols) + " |")
    for name, qs in baselines.items():
        lines.append(f"| {name} | " + " | ".join("" if qs.get(c) is None else f"{qs[c]:.3f}" for c in cols) + " |")
    lines += ["", "Retrieval minus each untuned pass, paired over the same 200 scans (95% interval):", "",
              head, "|" + "---|" * (len(cols) + 1)]
    for n, d in vs_untuned.items():
        lines.append(f"| vs {n} | " + " | ".join(f"{d[c]['diff']:+.3f} [{d[c]['lower']:+.3f}, {d[c]['upper']:+.3f}]"
                                                for c in cols) + " |")
    lines += ["", "Noise bar, untuned pass minus untuned pass:", "", head, "|" + "---|" * (len(cols) + 1)]
    for n, d in noise.items():
        lines.append(f"| {n} | " + " | ".join(f"{d[c]['diff']:+.3f} [{d[c]['lower']:+.3f}, {d[c]['upper']:+.3f}]"
                                             for c in cols) + " |")
    lines += ["", "Agreement with the neighbours' majority answer (share of scans with a majority):", "",
              "| row | " + " | ".join(k.replace("lesion_", "") for k in KEYS) + " |", "|" + "---|" * (len(KEYS) + 1)]
    for n, a in agree.items():
        lines.append(f"| {n} | " + " | ".join("" if a[k]["agree_share"] is None else
                                             f"{a[k]['agree_share']:.2f} (n={a[k]['with_majority']})" for k in KEYS) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nuntuned mean range {band}, largest untuned pair difference {widest:.4f}; retrieval mean "
          f"{scores['retrieval']['mean']:.4f}: "
          f"{'INSIDE the untuned noise (a result, not a failure: report it, do not tune)' if inside else 'DIFFERS from every untuned pass beyond the noise'}")
    print(f"written compare.json and table.md to {out}")
    return 0


def cmd_arms(args) -> int:
    """The example-CT arms (spec 2026-11-22-ct-findings-example-ct-arms.md): each pass scored on the VAL pool scans it
    answers (arm C answers only the 190 with organ ranges), every pass also on the scans all passes share, and each
    --pair X:Y as X minus Y paired over the scans both answer."""
    gold = {r["id"]: r for r in ad.read_split("val")}
    pool = set(ad.pools()["val"])
    passes = {}
    for item in args.passes:
        name, _, path = item.partition("=")
        passes[name] = load_answers(path)
        extra = set(passes[name]) - pool
        if extra:
            raise SystemExit(f"{name}: {len(extra)} answers outside the VAL pool")
    cover = {n: sorted(set(a) & pool) for n, a in passes.items()}
    shared = sorted(set.intersection(*(set(c) for c in cover.values())))
    scores = {n: aurocs(gold, a, cover[n]) for n, a in passes.items()}
    scores_shared = {n: aurocs(gold, a, shared) for n, a in passes.items()}
    errors = {n: sum(1 for i in cover[n] if a[i].get("error")) for n, a in passes.items()}
    pairs = {}
    for item in args.pair:
        x, _, y = item.partition(":")
        ids = sorted(set(cover[x]) & set(cover[y]))
        pairs[f"{x} - {y}"] = {"n": len(ids), **paired(gold, passes[x], passes[y], ids)}
    result = {"scans": {n: len(c) for n, c in cover.items()}, "shared_scans": len(shared), "scores": scores,
              "scores_on_shared": scores_shared, "program_errors": errors, "pairs": pairs, "bootstrap": BOOTSTRAP}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arms.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    cols = (*KEYS, "mean", "mean3")
    head = "| row | scans | " + " | ".join(c.replace("lesion_", "") for c in cols) + " |"
    rule = "|" + "---|" * (len(cols) + 2)
    lines = [head, rule]
    lines += [f"| {n} | {len(cover[n])} | " + " | ".join(f"{scores[n][c]:.3f}" for c in cols) + " |" for n in passes]
    lines += ["", f"Every pass on the {len(shared)} scans all of them answer:", "", head, rule]
    lines += [f"| {n} | {len(shared)} | " + " | ".join(f"{scores_shared[n][c]:.3f}" for c in cols) + " |"
              for n in passes]
    lines += ["", "Paired differences over the scans both passes answer (95% interval):", "", head, rule]
    for name, d in pairs.items():
        lines.append(f"| {name} | {d['n']} | " + " | ".join(
            f"{d[c]['diff']:+.3f} [{d[c]['lower']:+.3f}, {d[c]['upper']:+.3f}]" for c in cols) + " |")
    (out / "arms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"program errors {errors}; written arms.json and arms.md to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("usage")
    u.add_argument("answers")
    u.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--retrieval", required=True)
    c.add_argument("--untuned", action="append", default=[], help="name=answers.json, once per untuned pass")
    c.add_argument("--baselines", default="")
    c.add_argument("--out", required=True)
    m = sub.add_parser("arms")
    m.add_argument("--pass", dest="passes", action="append", required=True, help="name=answers.json, once per pass")
    m.add_argument("--pair", action="append", default=[], help="X:Y, scored as X minus Y")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"usage": cmd_usage, "compare": cmd_compare, "arms": cmd_arms}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
