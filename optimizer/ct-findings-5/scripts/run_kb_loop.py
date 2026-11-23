"""The knowledge-base pilot's builder/tester loop (spec planning/vista_eval_vlm/2026-11-23-ct-knowledge-base-pilot.md,
Approved 2026-11-23). Run with KB_ARM=1 from the engine's environment.

    init                         the work folder (run.py init), the builder's local copy of the source scans, pins
    pins [--write]               fingerprint the builder session; with --write commit its pin beside run.py's
    loop --rounds 5 [--tester-scans 100] [--run-id kb-r1] [--cap 200]        PAID
    usage ANSWERS_JSON [--out F]  $0: how a pass's sessions used the library (a VAL arm), from their transcripts
    report [--run-id kb-r1]      $0: per-round scores, library sizes, use of the library, spend

One round k:
  1. builder   a sealed Sonnet session (the engine's contained editing session) edits library/ of the latest version.
               It reads the source scans (TRAIN, never a tester patient) and, from round 2, the tester feedback of every
               earlier round. Its library must pass tools/kb.py check with every entry from a source scan, or the round
               stops. The edit is committed as program-v<k>.
  2. tester    one sealed session per scan (the program runner, as run.py val-pass) reads round k's TRAIN scans with
               program-v<k>; scored per question (AUROC) on TRAIN, which with KB_ARM=1 is the tester set.
  3. feedback  for the next builder: per scan its answer key, probabilities, reasons, findings note, the entries it
               opened and whether it went back to its own scan after; per entry how often it was opened on scans the
               tester got right vs wrong. Written to <work>/kb-tester/round-<k>/ (a read-only input next round).
The best round is the one whose tester batch scored the highest mean AUROC over the two targets (lymph nodes,
pneumonitis). VAL is never used here: score it once with run.py val-pass (scripts/run_kb_arms.sh).

Stops: total spend (tester sessions + builder sessions) would pass --cap before a step starts; a builder session fails
or leaves an invalid library. Resumes: a finished step of a round (its record in private/kb-loop/<run>/state.json) is
not redone. Prints counts, scores and dollars only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("KB_ARM", "1")

import config  # noqa: E402

if not config.KB_ARM:
    raise SystemExit("run with KB_ARM=1")

import adapters as ad  # noqa: E402
import optimizer as opt  # noqa: E402
import run  # noqa: E402
from engine.materialize import materialize  # noqa: E402
from engine.versioning import commit_new_version  # noqa: E402
from kb_usage import usage  # noqa: E402

KB_TOOLS = config.SEED_PROGRAM_DIR / "tools"
BUILDER_DIR = HERE / "kb_builder"
BUILDER_MODEL = os.environ.get("KB_BUILDER_MODEL", "claude-sonnet-5-5")  # DECIDE: Phil 2026-11-23, Sonnet both roles
BUILDER_BUDGET_USD = float(os.environ.get("KB_BUILDER_BUDGET_USD", "8"))
BUILDER_TIMEOUT_SECONDS = int(os.environ.get("KB_BUILDER_TIMEOUT_SECONDS", "3600"))
BUILDER_EFFORT = os.environ.get("KB_BUILDER_EFFORT", "high")
TARGETS = ("lesion_lymph_nodes", "pneumonitis")  # spec, Decided: the win is judged on these two
PIN_KIND = "builder"


def layout():
    return opt.Layout()


def source_dir(lay) -> Path:
    return lay.root / "kb-source"


def tester_dir(lay) -> Path:
    return lay.root / "kb-tester"


def state_path(lay, run_id: str) -> Path:
    return lay.private / "kb-loop" / run_id / "state.json"


def load_state(lay, run_id: str) -> dict:
    p = state_path(lay, run_id)
    return json.loads(p.read_text()) if p.is_file() else {"run_id": run_id, "rounds": {}}


def save_state(lay, state: dict) -> None:
    p = state_path(lay, state["run_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(p)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------------------------------


def builder_prompt(iter_n: int, paths) -> str:
    tester = paths.inputs["tester"]
    first = ("This is round 1: the library is empty and there is no tester feedback yet. Build a first library from "
             "the source scans." if iter_n == 1 else
             f"This is round {iter_n}. The tester has read rounds 1 to {iter_n - 1} with your earlier libraries; its "
             f"feedback is in `{tester}/round-<k>/`. Start with the latest round: what it got wrong, which entries it "
             "opened and whether they helped. Then revise the library.")
    return f"""{first}

## Where things are

- `{paths.inputs['task.md']}`: the task brief. Read it first.
- `{paths.inputs['source']}/`: the source scans you build entries from (README.md and index.json there).
- `{tester}/`: the tester's feedback, one folder per finished round (empty in round 1).
- `library/` in your working folder (`/workspace/rw/program`): the knowledge base. Edit files under `library/` only;
  `instructions.md` and `tools/` are the tester's and must not change (a change there refuses your whole edit).
- `tools/kb.py` and `tools/view_ct.py`: the tester's own tools. Render entry images with `python3 -B tools/kb.py render`.
- `{paths.inputs['notebook.md']}`: your notebook so far. Copy it to `{paths.outputs['notebook.md']}` first, then
  edit it there.
- `{paths.outputs['change_note.json']}`: write this round's change note here.
- `{paths.versions}/`: earlier versions of the program, as plain folders.

Images: every image you open stays in this session's memory, and the session fails once they add up past about 30 MB,
losing everything you built. So explore source scans at `--size 768` or less (view_ct.py and kb.py render both take
--size), check each entry image right after you render it with a preview at `--size 640` written to /tmp (then write
the entry's own image at the default size without opening it again), open each image once, and add at most 25 new
entries this round.

Before you finish, run `python3 -B tools/kb.py check` from your working folder and fix every problem it prints.
"""


def build_builder(lay, *, run_id: str, pin: str):
    from isolation.contained_agent import ContainedOptimizerAgent, DeclaredOutput

    import optimizer_prompt as op

    prof, credential = config.ROUTE.session_auth()
    token = None if config.ROUTE.is_vertex else run.load_token()[0]
    tester_dir(lay).mkdir(parents=True, exist_ok=True)
    return ContainedOptimizerAgent(
        model=BUILDER_MODEL,
        system_prompt=(BUILDER_DIR / "instructions.md").read_text(encoding="utf-8"),
        prompt_builder=builder_prompt,
        manifest=ad.TaskStore().manifest(),
        manifest_for_version=lambda sha: ad.TaskStore().manifest(),
        repo_root=lay.repo,
        materialize_root=lay.versions,
        iter_root=lay.repo / "iter",
        forbidden=opt.optimizer_forbidden(lay),
        profile=prof,
        credential=credential,
        image=run.images()[0],
        expected_fingerprint=pin,
        transcript_dir=lay.transcripts / "builder",
        readable_inputs=(("task.md", lay.inputs / "kb_task.md"), (op.NOTEBOOK_NAME, lay.notebook),
                         ("source", source_dir(lay)), ("tester", tester_dir(lay))),
        declared_outputs=(DeclaredOutput(op.CHANGE_NOTE_NAME), DeclaredOutput(op.NOTEBOOK_NAME, destination=lay.notebook)),
        notes_root=lay.notes,
        run_id=run_id,
        validation_roots=(lay.private, config.DATA_DIR),
        max_budget_usd=BUILDER_BUDGET_USD,
        effort=BUILDER_EFFORT,
        web_search=False,
        timeout_seconds=BUILDER_TIMEOUT_SECONDS,
        host_env={config.TOKEN_ENV_NAME: token} if token else None,
        env=opt.ENV,
        patient_data=config.PATIENT_DATA,
    )


def stage_builder_brief(lay) -> None:
    lay.inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILDER_DIR / "task.md", lay.inputs / "kb_task.md")


def library_problems(tree: Path) -> list[str]:
    """tools/kb.py's own check over a version's library, plus: every entry's source is a source scan."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kb_check", tree / "tools" / "kb.py")
    kb = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(tree / "tools"))
    try:
        spec.loader.exec_module(kb)
        kb.LIBRARY = tree / "library"
        kb.ENTRIES = kb.LIBRARY / "entries"
        sources = {s["id"] for s in json.loads((config.KB_DIR / "source.json").read_text())}
        return [f"{eid}: {p}" for eid, e in kb.entries() for p in kb.entry_problems(eid, e, sources)]
    finally:
        sys.path.remove(str(tree / "tools"))
        sys.modules.pop("view_ct", None)


def library_stats(tree: Path) -> dict:
    rows = [json.loads(f.read_text()) for f in sorted((tree / "library" / "entries").glob("*/entry.json"))]
    return {"entries": len(rows), "text_only": sum(bool(r.get("text_only")) for r in rows),
            "by_finding": dict(Counter(r.get("finding") for r in rows)),
            "by_answer": dict(Counter(f"{r.get('finding')}:{r.get('answer', '-')}" for r in rows))}


# --------------------------------------------------------------------------------------------------
# The tester and its feedback
# --------------------------------------------------------------------------------------------------


def rates() -> dict[str, float]:
    rows = ad.read_split("train")
    out = {}
    for key, _t, _q in config.QUESTIONS:
        s = [r[key] for r in rows if r.get(key) is not None]
        out[key] = sum(s) / len(s) if s else 0.5
    return out


def word(v) -> str:
    return "yes" if v == 1 else "no" if v == 0 else "not scorable"


def write_feedback(lay, k: int, artifacts, score) -> dict:
    """The tester's round k as the builder will read it; returns the round's usage summary."""
    gold = {r["id"]: r for r in ad.read_split("train")}
    answers = ad._answers(artifacts)
    rate = rates()
    out = tester_dir(lay) / f"round-{k}"
    if out.exists():
        shutil.rmtree(out)
    (out / "findings").mkdir(parents=True)
    scans, entry_use = [], defaultdict(lambda: Counter())
    for i, a in sorted(answers.items()):
        events = next((f for f in ad._trace_files(artifacts, i) if f.name == "events.jsonl"), None)
        u = usage(events)
        right = {key: (None if gold[i].get(key) is None else
                       (not a["error"]) and ((ad.prob(a, key) >= rate[key]) == (gold[i][key] == 1)))
                 for key, _t, _q in config.QUESTIONS}
        note = Path(a["findings_log"]) if a.get("findings_log") else None
        if note and note.is_file():
            shutil.copy2(note, out / "findings" / f"{i}.md")
        scans.append({"scan": i, "answer_key": {key: word(gold[i].get(key)) for key, _t, _q in config.QUESTIONS},
                      "probability": {key: (None if a["error"] else round(100 * ad.prob(a, key))) for key, _t, _q in config.QUESTIONS},
                      "right_side": right, "error": a["error"], "reasons": a.get("reasons") or [],
                      "findings_note": f"findings/{i}.md" if (out / "findings" / f"{i}.md").is_file() else None, **u})
        for eid in u["entries_opened"]:
            entry_use[eid]["opened"] += 1
            for key in TARGETS:
                if right[key] is not None:
                    entry_use[eid][f"{key}:{'right' if right[key] else 'wrong'}"] += 1
    per_q = run.per_question(score)
    summary = {
        "round": k, "scans": len(scans),
        "auroc": {key: per_q[key]["auroc"] for key, _t, _q in config.QUESTIONS},
        "target_mean_auroc": round(sum(per_q[t]["auroc"] for t in TARGETS) / len(TARGETS), 4),
        "right_side_means": "a probability on the correct side of that question's yes rate in the tester set "
                            f"({', '.join(f'{key} {100 * rate[key]:.0f}%' for key, _t, _q in config.QUESTIONS)})",
        "share_opened_an_entry": round(sum(bool(s["entries_opened"]) for s in scans) / max(len(scans), 1), 3),
        "mean_entries_opened": round(sum(len(s["entries_opened"]) for s in scans) / max(len(scans), 1), 2),
        "share_went_back_to_own_scan": round(sum(s["own_views_after_an_open"] > 0 for s in scans if s["entries_opened"])
                                             / max(sum(bool(s["entries_opened"]) for s in scans), 1), 3),
        "failed_sessions": sum(bool(s["error"]) for s in scans),
    }
    (out / "scans.json").write_text(json.dumps(scans, indent=1))
    (out / "entry_use.json").write_text(json.dumps({e: dict(c) for e, c in sorted(entry_use.items())}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "README.md").write_text(
        f"# Tester, round {k}\n\nThe tester read {len(scans)} TRAIN scans (other patients than the source scans) with "
        f"program-v{k}'s library.\n\n- summary.json: AUROC per question on these scans, and how much the library was used.\n"
        "- scans.json: per scan, the answer key, the tester's probabilities (0-100), whether each is on the right side "
        "of the question's yes rate, its stated reasons, the entries it opened, how many example images it read, and "
        "how often it went back to its own scan after opening one.\n- findings/<scan>.md: the tester's own findings note.\n"
        "- entry_use.json: per entry, how often it was opened, and on how many of those scans each target question "
        "came out on the right vs the wrong side.\n\nThe tester's scans are not available to you: only what it wrote.\n")
    for root, _d, files in os.walk(out):
        for f in files:
            os.chmod(Path(root) / f, 0o444)
    return summary


def pass_cost(run_id: str) -> tuple[float, int]:
    """Dollars and sessions of every finished tester pass under this pass run id, from the runner's receipts."""
    total, n = 0.0, 0
    for receipts in (layout().program_outputs / "archive").glob("*/receipts"):
        for f in receipts.glob("*.json"):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if r.get("run_id") == run_id:
                total += sum(float(s.get("cost_usd") or 0.0) for s in r.get("stages", []))
                n += 1
    return round(total, 4), n


def spent(state: dict) -> float:
    return round(sum(float(r.get("builder_usd") or 0) + float(r.get("tester_usd") or 0)
                     for r in state["rounds"].values()), 4)


# --------------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------------


def cmd_init(_args) -> int:
    lay = layout()
    if not lay.repo.exists():
        run.init_layout(lay)
    stage_builder_brief(lay)
    if not (source_dir(lay) / "index.json").is_file():
        subprocess.run([sys.executable, "-B", str(HERE / "scripts" / "build_kb_data.py"), "stage", str(source_dir(lay))],
                       check=True)
    tester_dir(lay).mkdir(parents=True, exist_ok=True)
    if not (lay.repo / run.PIN_FILE).exists():  # run.py's pins (program, optimizer, search): run.py val-pass needs them
        run.cmd_pins(argparse.Namespace(write=True))
    print(f"work folder {lay.root}; source {source_dir(lay)}")
    return 0


def cmd_pins(args) -> int:
    from isolation.setup_fingerprint import PIN_KEY, UNPINNED, fingerprint_hash, platform_key, setup_fingerprint

    lay = layout()
    stage_builder_brief(lay)
    agent = build_builder(lay, run_id="pins", pin=UNPINNED)
    value = fingerprint_hash(setup_fingerprint(agent.setup_description()))
    print(f"builder pin on {platform_key()}: {value}")
    if not args.write:
        return 0
    path = lay.repo / run.PIN_FILE
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault("runtime_pins", {}).setdefault(PIN_KEY, {}).setdefault(PIN_KIND, {})[platform_key()] = value
    path.write_text(json.dumps(data, indent=2) + "\n")
    run.git(lay.repo, "add", run.PIN_FILE)
    run.git(lay.repo, "commit", "-m", f"builder pin for {platform_key()}")
    print(f"committed {path}")
    return 0


def builder_pin(lay) -> str:
    from isolation.setup_fingerprint import committed_pin

    pin = committed_pin(lay.repo, run.PIN_FILE, kind=PIN_KIND)
    if pin is None:
        raise SystemExit("no committed builder pin: run scripts/run_kb_loop.py pins --write")
    return pin


def cmd_loop(args) -> int:
    lay = layout()
    state = load_state(lay, args.run_id)
    rounds = json.loads((config.KB_DIR / "rounds.json").read_text())
    pins = run.committed_pins(lay)
    bpin = builder_pin(lay)
    per_scan = float(os.environ.get("KB_PER_SCAN_ESTIMATE_USD", "0.25"))  # until a round has measured it
    stage_builder_brief(lay)
    opt.stage_files(lay)
    def total() -> float:  # this run's spend plus what the launcher already spent (the cost check)
        return spent(state) + args.already_spent

    def reserve() -> tuple[float, float]:  # the two VAL arms still to come, at the measured (or estimated) cost per scan
        measured = [r for r in state["rounds"].values() if r.get("tester_sessions")]
        est = (sum(r["tester_usd"] for r in measured) / sum(r["tester_sessions"] for r in measured)) if measured else per_scan
        return est, est * args.val_scans

    for k in range(1, args.rounds + 1):
        rec = state["rounds"].setdefault(str(k), {"round": k})
        # 1. the builder
        if "tag" not in rec:
            if total() + BUILDER_BUDGET_USD + reserve()[1] > args.cap:
                print(f"STOPPED: round {k}'s builder could pass the ${args.cap:.0f} cap with the VAL arms "
                      f"(~${reserve()[1]:.2f}) kept back (spent ${total():.2f})")
                return 3
            base = state["rounds"][str(k - 1)]["tag"] if k > 1 else "program-v0"
            sha = run.git(lay.repo, "rev-parse", f"{base}^{{commit}}").strip()
            # The work repo's library back to the base: a refused earlier attempt may have left its files committed.
            shutil.rmtree(lay.repo / "library", ignore_errors=True)
            run.git(lay.repo, "checkout", sha, "--", "library")
            tree = lay.versions / f"kb-{args.run_id}-round{k}-base-{base}"
            if not tree.exists():
                materialize(ad.TaskStore().manifest(), sha, repo_root=lay.repo, dest=tree)
            agent = build_builder(lay, run_id=args.run_id, pin=bpin)
            for attempt in (1, 2):  # a failed session keeps nothing; one retry before the round stops
                if attempt == 2 and total() + BUILDER_BUDGET_USD + reserve()[1] > args.cap:
                    break
                print(f"round {k}: builder on {base} (attempt {attempt}) ...", flush=True)
                outcome = agent.run(iter_n=k, materialized_path=tree)
                rec["builder_usd"] = round(float(rec.get("builder_usd") or 0.0) + float(outcome.cost_usd or 0.0), 4)
                rec["builder_attempts"] = int(rec.get("builder_attempts") or 0) + 1
                rec["builder_detail"] = str(outcome.detail)[:300]
                save_state(lay, state)
                if outcome.exit_code == 0:
                    break
                print(f"round {k}: builder attempt {attempt} failed: {str(outcome.detail)[:200]}", flush=True)
            if outcome.exit_code != 0:
                print(f"STOPPED: round {k}'s builder failed: {outcome.detail}")
                return 1
            tag, new_sha = commit_new_version(ad.TaskStore().manifest(), f"kb {args.run_id} round {k}", repo_root=lay.repo,
                                              allow_empty=True)
            vtree = lay.versions / f"kb-{args.run_id}-round{k}-{tag}"
            materialize(ad.TaskStore().manifest(), new_sha, repo_root=lay.repo, dest=vtree)
            problems = library_problems(vtree)
            if problems:
                rec.setdefault("refused_tags", []).append(tag)
                save_state(lay, state)
                print(f"STOPPED: {tag}'s library has {len(problems)} problems, first: {problems[:3]}; re-running "
                      f"starts round {k} again from {base}")
                return 1
            rec.update({"tag": tag, "version_sha": new_sha, "library": library_stats(vtree)})
            save_state(lay, state)
            print(f"round {k}: {tag} library {rec['library']}; builder ${rec['builder_usd']:.2f}", flush=True)
        # 2. the tester
        if "scores" not in rec:
            ids = sorted(rounds[str(k)])[: args.tester_scans]
            est, keep = reserve()
            if total() + est * len(ids) + keep > args.cap:
                print(f"STOPPED: round {k}'s tester pass (~${est * len(ids):.2f}) could pass the ${args.cap:.0f} cap "
                      f"with the VAL arms (~${keep:.2f}) kept back (spent ${total():.2f})")
                return 3
            tag = rec["tag"]
            tree = lay.versions / f"kb-{args.run_id}-round{k}-{tag}"
            if not tree.exists():
                materialize(ad.TaskStore().manifest(), rec["version_sha"], repo_root=lay.repo, dest=tree)
            inputs = ad.TaskSource(lay.private / "kb-batches").stage(ids, "train", f"kb-{args.run_id}-round{k}-n{len(ids)}")
            pass_run = f"kb-{args.run_id}-round{k}"
            print(f"round {k}: tester on {len(ids)} TRAIN scans with {tag} ...", flush=True)
            runner = run.program_runner(lay, image=run.images()[1], pin=pins["program"])
            with runner:
                artifacts = runner.as_runner(run_id=pass_run).run(tree, inputs)
            rec["tester_usd"], rec["tester_sessions"] = pass_cost(pass_run)
            save_state(lay, state)
            if artifacts.status != "ok":
                print(f"STOPPED: round {k}'s tester pass failed: {artifacts.status} {artifacts.error}")
                return 1
            score = ad.TaskScorer().score(artifacts, "train", {})
            summary = write_feedback(lay, k, artifacts, score)
            rec.update({"answers": artifacts.artifact_refs[0].path, "scores": summary["auroc"],
                        "target_mean_auroc": summary["target_mean_auroc"],
                        "usage": {k2: summary[k2] for k2 in ("share_opened_an_entry", "mean_entries_opened",
                                                             "share_went_back_to_own_scan", "failed_sessions")},
                        "finished": now()})
            save_state(lay, state)
            print(f"round {k}: targets {summary['target_mean_auroc']:.3f} {summary['auroc']}; tester "
                  f"${rec['tester_usd']:.2f} for {rec['tester_sessions']}; spent ${total():.2f}", flush=True)
    done = [r for r in state["rounds"].values() if "target_mean_auroc" in r]
    best = max(done, key=lambda r: (r["target_mean_auroc"], -r["round"]))
    state["best"] = {"round": best["round"], "tag": best["tag"], "target_mean_auroc": best["target_mean_auroc"]}
    state["spent_usd"] = spent(state)
    save_state(lay, state)
    print(json.dumps({"best": state["best"], "spent_usd": state["spent_usd"]}))
    return 0


def trace_events(answers: Path, call_id: str) -> Path | None:
    """A pass's session transcript for one scan, latest attempt (as adapters._trace_files): beside its answers.json."""
    root = Path(answers).parent / "transcripts" / call_id
    attempts = sorted(root.glob("attempt*"), key=lambda p: int(p.name.removeprefix("attempt") or 0))
    runs = sorted(attempts[-1].glob("*/*")) if attempts else []
    f = runs[-1] / "events.jsonl" if runs else None
    return f if f is not None and f.is_file() else None


def cmd_usage(args) -> int:
    """How one pass's sessions used the library (a VAL arm's answers.json, on the machine that ran it)."""
    answers = json.loads(Path(args.answers).read_text())["answers"]
    rows = {i: usage(trace_events(Path(args.answers), i)) for i in sorted(answers)}
    opened = [r for r in rows.values() if r["entries_opened"]]
    summary = {"scans": len(rows), "with_transcript": sum(r["transcript"] for r in rows.values()),
               "share_opened_an_entry": round(len(opened) / max(len(rows), 1), 3),
               "mean_entries_opened": round(sum(len(r["entries_opened"]) for r in rows.values()) / max(len(rows), 1), 2),
               "mean_example_images_read": round(sum(r["example_images_read"] for r in rows.values()) / max(len(rows), 1), 2),
               "share_went_back_to_own_scan": round(sum(r["own_views_after_an_open"] > 0 for r in opened) / max(len(opened), 1), 3),
               "entries_opened_at_least_once": len({e for r in rows.values() for e in r["entries_opened"]})}
    print(json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    return 0


def cmd_report(args) -> int:
    state = load_state(layout(), args.run_id)
    rows = []
    for k in sorted(state["rounds"], key=int):
        r = state["rounds"][k]
        rows.append({x: r.get(x) for x in ("round", "tag", "library", "target_mean_auroc", "scores", "usage",
                                           "builder_usd", "tester_usd", "tester_sessions")})
    print(json.dumps({"run_id": args.run_id, "rounds": rows, "best": state.get("best"), "spent_usd": spent(state)}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"rounds": rows, "best": state.get("best"), "spent_usd": spent(state)}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("pins")
    p.add_argument("--write", action="store_true")
    lp = sub.add_parser("loop")
    lp.add_argument("--rounds", type=int, default=5)
    lp.add_argument("--tester-scans", type=int, default=100)
    lp.add_argument("--run-id", default="kb-r1", type=run.run_id_arg)
    lp.add_argument("--cap", type=float, default=200.0, help="the pilot's total, VAL arms included (spec)")
    lp.add_argument("--val-scans", type=int, default=380, help="VAL sessions to keep room for: 2 arms x 190")
    lp.add_argument("--already-spent", type=float, default=0.0, help="dollars spent before this run (the cost check)")
    up = sub.add_parser("usage")
    up.add_argument("answers")
    up.add_argument("--out")
    rp = sub.add_parser("report")
    rp.add_argument("--run-id", default="kb-r1")
    rp.add_argument("--out")
    args = ap.parse_args(argv)
    return {"init": cmd_init, "pins": cmd_pins, "loop": cmd_loop, "usage": cmd_usage, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
