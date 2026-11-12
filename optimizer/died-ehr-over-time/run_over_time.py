#!/usr/bin/env python3
"""Run the death agent over time: every decision point in order, both arms, Haiku 5.5, on the engine's sealed
passes. No optimizer: the two instruction files are fixed (program_fresh/, program_memory/).

    run_over_time.py pins [--write]          fingerprint this task's session setup and keep it in the work folder
    run_over_time.py leak-check              $0: every staged record, dated items after its row's cut, per point
    run_over_time.py run --run-id R          the paid pass (Expensive: needs Phil's go); resume = the same line
    run_over_time.py run --run-id R --dry-run   $0: the same driver over fake sessions (no Docker, no Claude)
    run_over_time.py score --run-id R [--out F]   $0: the score tables from the run's answers
    run_over_time.py print-prompt --point m6 --arm memory    one prompt, with made-up earlier answers

How it uses the engine without changing it. Each (point, arm) is cut into passes of common.CHUNK calls; each pass is
an ordinary ``ProgramRunner`` pass of that arm's instruction folder over a staged batch file listing its calls
(``program.program_spec``), the same call the died-2yr-ehr task's ``score-val`` makes. Points run in order, and
within a point the fresh arm first. Before the memory arm's passes at point k, the driver reads that arm's parsed
answers at every earlier point (the engine's answers.json of each finished pass) and writes each row's memory block
into the batch, so it is part of the call and of the pass key. At treatment start there is no earlier answer, so the
memory arm's prompt would equal the fresh arm's: the memory arm takes the fresh arm's treatment-start answers (one
set of calls, not two). A finished pass is served from the engine's archive at $0, so re-running the same line after
a network stop resumes where it stopped.

Dollar stops, checked before every pass from the engine's receipts and killed attempts (common.RUN_BUDGET_USD, the
died-2yr-ehr setting's name): a pass starts only if spend so far + its calls x the per-call cap stays within $250, so
the cap cannot be passed; and once 100 calls are paid, stop if they averaged over $0.10 a call. Leak stops: a record
with anything dated after its row's cut stops the run before any call sees it. An earlier reason in a memory block
that names a date after the current cut is dropped from the block and counted per point (Phil 2026-11-11); anything
left after that stops the run. Treatment-start check (spec): when the fresh arm's treatment-start answers are in, and
before any later point is paid for, its mean AUROC is compared with the Haiku probability seed (score.start_gate);
more than 0.05 below stops the run to ask Phil, and OVER_TIME_PAST_START_GATE=1 on the same line goes on past it.

Exit codes: 0 finished, 3 stopped by a rule (the reason is printed), 1 anything else.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as c  # noqa: E402  (first: sets died-2yr-ehr's settings, then imports its config)
import config  # noqa: E402
import optimizer as opt  # noqa: E402  (died-2yr-ehr's Layout and forbidden list)
import program  # noqa: E402
import score as sc  # noqa: E402

from engine.schemas import RunInputs  # noqa: E402
from isolation.session_scope import Forbidden  # noqa: E402

STOPPED = 3


class Stop(Exception):
    """A rule stopped the run (budget, cost per call, leak)."""


def layout() -> "opt.Layout":
    return opt.Layout(root=c.WORKDIR)


def forbidden(lay) -> tuple:
    """died-2yr-ehr's list for this work folder (its code, the work folder's private/, the home Claude and gcloud
    folders, the whole mount, the local data copy), plus this task's own code and labels."""
    return (*opt.forbidden(lay), Forbidden(HERE, representatives=("score.py", "TASK_NOTES.md")))


def trees(lay) -> dict[str, Path]:
    """Each arm's instruction folder, copied read-only under the work folder's versions/ (the runner's
    materialize_root). Re-copied only when the source changed."""
    out = {}
    for arm, src in program.PROGRAM_DIRS.items():
        dest = lay.versions / f"over-time-{arm}"
        same = dest.exists() and all((dest / f.relative_to(src)).is_file()
                                     and (dest / f.relative_to(src)).read_bytes() == f.read_bytes()
                                     for f in src.rglob("*") if f.is_file())
        if not same:
            if dest.exists():
                for p in [dest, *dest.rglob("*")]:
                    p.chmod(0o755 if p.is_dir() else 0o644)
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            for p in [*dest.rglob("*"), dest]:
                p.chmod(0o555 if p.is_dir() else 0o444)
        out[arm] = dest
    return out


def make_runner(lay, *, pin: str, image: str, fake=None, cls=None):
    """The engine's ProgramRunner over this task's spec, on died-2yr-ehr's route and image. ``fake``: the dry run's
    stand-in sessions (the engine's own test hooks), so nothing reaches Docker or Claude."""
    from isolation import sealed_session as ss
    from isolation.program_runner import ProgramRunner
    from isolation.session_scope import SessionScope

    import agentic

    fb = forbidden(lay)
    profile, credential = (ss.AnthropicApiProfile(token_env_name=config.TOKEN_ENV_NAME), ss.NO_CREDENTIAL) if fake \
        else config.ROUTE.session_auth()
    template = ss.SessionDescription(
        scope=SessionScope(program=lay.versions, denied=fb), posture="deny-by-default", tools=agentic.TOOLS,
        allowed_tools=agentic.TOOLS, model=config.PROGRAM_MODEL, image=image, profile=profile, credential=credential,
        transcript_dir=lay.transcripts / "program", expected_fingerprint=pin)
    spec = program.program_spec(fb)
    if cls is None:
        cls = ProgramRunner  # the climb passes its subclass (climb.ClimbRunner)
    if fake:
        from dataclasses import replace
        spec = replace(spec, token_env_name=None)
        return cls(spec=spec, template=template, output_root=lay.program_outputs,
                   materialize_root=lay.versions, _session_factory=fake.session,
                   _stack_factory=fake.stack, _docker=fake)
    return cls(spec=spec, template=template, output_root=lay.program_outputs, materialize_root=lay.versions)


# ==================================================================================================
# Spend
# ==================================================================================================


_RECEIPTS: dict = {}


def spend(lay, run_id: str) -> tuple[float, int]:
    """(dollars, paid calls) this run id has spent: every receipt (finished or failed call) plus every killed or
    interrupted attempt in every pass record, live, archived or set aside. An attempt whose cost was never read
    counts at the per-call cap."""
    root = lay.program_outputs
    dollars, calls, seen, seen_pass = 0.0, 0, set(), set()
    for f in root.rglob("receipts/*.json"):
        got = _RECEIPTS.get(f)
        if got is None:  # a receipt never changes once written: read each once per process
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            got = _RECEIPTS[f] = (r.get("run_id"), (r.get("pass_id"), r.get("call_id"), r.get("attempt")),
                                  sum(float(s.get("cost_usd") or 0.0) for s in r.get("stages", [])))
        rid, key, cost = got
        if rid != run_id or key in seen:
            continue
        seen.add(key)
        dollars += cost
        calls += 1
    for f in root.rglob("pass.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        key = (rec.get("pass_id"), rec.get("attempt"))
        if rec.get("run_id") != run_id or key in seen_pass:
            continue
        seen_pass.add(key)
        dollars += sum(float(a.get("cost_usd") or 0.0) for a in rec.get("killed_attempts", []))
        dollars += int(rec.get("cost_unknown_attempts") or 0) * config.PROGRAM_CALL_BUDGET_USD
    return dollars, calls


def check_spend(lay, run_id: str, n_next: int, state: dict, *, cap: float | None = None, reserve: float = 0.0) -> None:
    """Stop before a pass that could take spend past ``cap`` (default common.RUN_BUDGET_USD): patient sessions so
    far + ``reserve`` (the climb's optimizer allowance) + every call of the pass at its cap. Then the
    first-100-calls cost check."""
    cap = c.RUN_BUDGET_USD if cap is None else cap
    dollars, calls = spend(lay, run_id)
    worst = dollars + reserve + n_next * config.PROGRAM_CALL_BUDGET_USD
    if worst > cap:
        raise Stop(f"spent ${dollars:.2f} over {calls} calls (+ ${reserve:.2f} kept for the optimizer); the next pass "
                   f"({n_next} calls at up to ${config.PROGRAM_CALL_BUDGET_USD:.2f} each) could pass the ${cap:.0f} "
                   f"hard stop")
    if calls >= c.EARLY_CALLS and not state.get("early_checked"):
        state["early_checked"] = True
        per = dollars / calls
        print(f"cost check after the first {calls} calls: ${per:.4f} a call (stop above ${c.EARLY_MAX_COST_PER_CALL:.2f})")
        if per > c.EARLY_MAX_COST_PER_CALL:
            raise Stop(f"the first {calls} calls cost ${per:.4f} each, over ${c.EARLY_MAX_COST_PER_CALL:.2f}: stop and ask")


# ==================================================================================================
# Leak checks
# ==================================================================================================


_SCANNED: dict[tuple[str, str], dict] = {}


def record_leaks(point: str, split: str = "val", patients: set | None = None) -> tuple[int, dict]:
    """Records of one point's rows with anything dated after the row's cut: (rows checked, counts by kind). Each
    (record, cut) is scanned once per process."""
    total: dict[str, int] = {}
    pr = program.point_rows(point, split, patients)
    for r in pr:
        key = (r["record"], r["cut"])
        if key not in _SCANNED:
            _SCANNED[key] = program.record_violations(config.DATA_DIR / "patients" / r["record"], r["cut"])
        for k, n in _SCANNED[key].items():
            total[k] = total.get(k, 0) + n
    return len(pr), total


def cmd_leak_check(args) -> int:
    bad = 0
    for split in (("val", "train") if args.split == "all" else (args.split,)):
        for point in c.POINT_KEYS:
            n, v = record_leaks(point, split)
            bad += sum(v.values())
            print(f"record leak check {split} {point}: {n} records, {sum(v.values())} violations {v or ''}")
    print("PASS" if not bad else "FAIL")
    return 1 if bad else 0


# ==================================================================================================
# The driver
# ==================================================================================================


def answers_dir(lay, run_id: str) -> Path:
    return lay.private / "runs" / run_id / "answers"


def drive(lay, runner, run_id: str, *, chunk: int = c.CHUNK) -> int:
    from isolation.pass_outputs import PassStopped

    tr = trees(lay)
    out = answers_dir(lay, run_id)
    out.mkdir(parents=True, exist_ok=True)
    lay.batches.mkdir(parents=True, exist_ok=True)
    state: dict = {}
    answers: dict[str, dict[str, dict]] = {}  # "<point>-<arm>" -> row -> answers.json entry
    with runner:
        bound = runner.as_runner(run_id=run_id)
        for point in c.POINT_KEYS:
            pr = program.point_rows(point)
            n, v = record_leaks(point)
            print(f"[{point}] record leak check: {n} records, {sum(v.values())} violations {v or ''}", flush=True)
            if v:
                raise Stop(f"a record at {point} holds something dated after its cut: stop")
            for arm in program.ARMS:
                name = f"{point}-{arm}"
                if arm == "memory" and point == c.POINT_KEYS[0]:
                    # No earlier answer exists here, so the memory arm's block would be empty; it takes the fresh
                    # arm's answers rather than paying for a second set (its instructions file differs only in
                    # describing that block). Memory minus fresh is therefore 0 here by construction.
                    answers[name] = answers[f"{point}-fresh"]
                    (out / f"{name}.json").write_text(json.dumps({"answers": answers[name], "shared_with": "fresh"}))
                    print(f"[{name}] shares the fresh arm's {len(answers[name])} answers (no earlier point)")
                    continue
                earlier = {p: answers[f"{p}-{arm}"] for p in c.POINT_KEYS[: c.POINT_KEYS.index(point)]}
                calls = []
                leaks = dropped = 0
                for r in pr:
                    mem, gone = program.memory_for(r, earlier) if arm == "memory" else ([], 0)
                    dropped += gone
                    leaks += program.memory_violations(r, mem)
                    calls.append({"row": r["row"], "record": r["record"], "memory": mem})
                if arm == "memory":
                    print(f"[{name}] memory leak check: {len(calls)} blocks, "
                          f"{sum(1 for x in calls if x['memory'])} with earlier answers, {dropped} earlier reasons dropped for a date "
                          f"after the cut, {leaks} violations", flush=True)
                    if leaks:
                        raise Stop(f"a memory block at {point} names a date after its row's cut: stop")
                got: dict[str, dict] = {}
                for i in range(0, len(calls), chunk):
                    part = calls[i:i + chunk]
                    check_spend(lay, run_id, len(part), state)
                    label = f"{name}-c{i // chunk:03d}"
                    path = lay.batches / f"{run_id}-{label}.json"
                    path.write_text(json.dumps({"arm": arm, "point": point, "calls": part}), encoding="utf-8")
                    inputs = RunInputs(input_ref=str(path), batch_id=label, split="val")
                    try:
                        arts = bound.run(tr[arm], inputs)
                    except PassStopped as exc:
                        print(f"STOPPED (pass_stopped): {exc}\nre-run the same line to resume")
                        return 2
                    got.update(json.loads(Path(arts.artifact_refs[0].path).read_text(encoding="utf-8"))["answers"])
                answers[name] = got
                (out / f"{name}.json").write_text(json.dumps({"answers": got}), encoding="utf-8")
                if name == f"{c.POINT_KEYS[0]}-fresh":
                    gate = start_gate(got)
                    sc.print_gate(gate)
                    (out.parent / "start_gate.json").write_text(json.dumps(gate, indent=1))
                    if not gate["pass"]:
                        if os.environ.get("OVER_TIME_PAST_START_GATE") != "1":
                            raise Stop("the fresh arm at treatment start is more than 0.05 AUROC below the Haiku "
                                       "probability seed: stop and ask Phil. To go on past this check after his go, "
                                       "re-run the same line with OVER_TIME_PAST_START_GATE=1 (finished passes are "
                                       "not re-run)")
                        print("treatment-start check failed; going on because OVER_TIME_PAST_START_GATE=1")
                failed = sum(1 for a in got.values() if a.get("error"))
                dollars, paid = spend(lay, run_id)
                print(f"[{name}] {len(got)} answers, {failed} failed; run so far ${dollars:.2f} over {paid} calls",
                      flush=True)
    dollars, paid = spend(lay, run_id)
    print(f"FINISHED: {sum(len(a) for k, a in answers.items() if not k.endswith('start-memory'))} calls answered; "
          f"${dollars:.2f} over {paid} paid calls (${dollars / max(paid, 1):.4f} a call)")
    return 0


#: Where the Haiku probability seed's zero-shot VAL answers sit on phil-sllm-02 (prob-summary-haiku-r1's work
#: folder; its round-1 VAL pass of the seed is named iter-1-val-<hash>). OVER_TIME_SEED_ANSWERS overrides.
SEED_GLOB = ".cache/alo-tasks/died-2yr-ehr-prob-summary-haiku/program-outputs/archive/iter-1-val-*/answers.json"


def seed_answers_path() -> Path | None:
    if os.environ.get("OVER_TIME_SEED_ANSWERS"):
        return Path(os.environ["OVER_TIME_SEED_ANSWERS"])
    import re as _re
    found = [p for p in Path.home().glob(SEED_GLOB) if _re.fullmatch(r"iter-1-val-[0-9a-f]{8}", p.parent.name)]
    return found[0] if len(found) == 1 else None


def start_gate(fresh_start: dict) -> dict:
    labels_path = Path(os.environ.get("OVER_TIME_SEED_LABELS", c.DIED_2YR_EHR_SPLITS / "val.jsonl"))
    seed, gold, note = sc.load_seed(seed_answers_path(), labels_path)
    print(f"Haiku seed reference: {note}")
    ids = {}
    if (c.DATA_DIR / "seed_ids.jsonl").is_file():
        ids = {r["patient"]: r["task_id"] for r in map(json.loads, (c.DATA_DIR / "seed_ids.jsonl").read_text().splitlines())}
    return {**sc.start_gate(fresh_start, program.split_rows("val"), program.labels(), seed, gold, ids), "seed_note": note}


def cmd_pins(args) -> int:
    from isolation.setup_fingerprint import UNPINNED, fingerprint_hash, platform_key, setup_fingerprint
    import run as d2  # died-2yr-ehr's run.py: its image names

    lay = layout()
    lay.versions.mkdir(parents=True, exist_ok=True)
    runner = make_runner(lay, pin=UNPINNED, image=d2.images()[1])
    fp = setup_fingerprint(runner.setup_description())
    h = fingerprint_hash(fp)
    print(json.dumps(fp, indent=2))
    print(f"\nplatform {platform_key()}: program {h}")
    if args.write:
        path = c.WORKDIR / "pins.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        data[platform_key()] = h
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"written {path}")
    return 0


def cmd_run(args) -> int:
    lay = layout()
    for d in (lay.versions, lay.private, lay.transcripts, lay.program_outputs):
        d.mkdir(parents=True, exist_ok=True)
    if not (lay.private / "README.txt").exists():
        (lay.private / "README.txt").write_text("Run records. Never mounted into a container.\n")
    try:
        if args.dry_run:
            import fake_sessions

            from isolation.setup_fingerprint import UNPINNED

            runner = make_runner(lay, pin=UNPINNED, image="alo-dry-run:0.0.0@sha256:" + "0" * 64,
                                 fake=fake_sessions.World())
        else:
            from isolation.setup_fingerprint import platform_key
            import run as d2  # died-2yr-ehr's run.py: image names and the seat token

            pins = json.loads((c.WORKDIR / "pins.json").read_text()) if (c.WORKDIR / "pins.json").exists() else {}
            if platform_key() not in pins:
                print(f"no pin for {platform_key()}: run run_over_time.py pins --write first")
                return 1
            if not config.ROUTE.is_vertex:
                token, _ = d2.load_token()
                if not token:
                    print("no Claude token for the seat route (or set ALO_ROUTE=vertex)")
                    return 1
            runner = make_runner(lay, pin=pins[platform_key()], image=d2.images()[1])
        return drive(lay, runner, args.run_id, chunk=args.chunk)
    except Stop as exc:
        print(f"STOPPED (rule): {exc}")
        return STOPPED


def load_answers(lay, run_id: str) -> dict[str, dict[str, dict]]:
    out = {}
    for f in sorted(answers_dir(lay, run_id).glob("*.json")):
        out[f.stem] = json.loads(f.read_text(encoding="utf-8"))["answers"]
    return out


def load_probes() -> dict:
    import pyarrow.parquet as pq

    t = pq.read_table(c.DATA_DIR / "probes.parquet").to_pylist()
    return {(r["row"], r["probe"], r["field"]): r["score"] for r in t}


def cmd_score(args) -> int:
    lay = layout()
    ans = load_answers(lay, args.run_id)
    if not ans:
        print(f"no answers for run {args.run_id} in {answers_dir(lay, args.run_id)}")
        return 1
    out = sc.score(ans, program.split_rows("val"), program.labels(), load_probes(), n_boot=args.n_boot)
    if f"{c.POINT_KEYS[0]}-fresh" in ans:
        out["start_gate"] = start_gate(ans[f"{c.POINT_KEYS[0]}-fresh"])
        sc.print_gate(out["start_gate"])
    sc.print_tables(out)
    dollars, paid = spend(lay, args.run_id)
    out["cost"] = {"dollars": round(dollars, 2), "paid_calls": paid, "per_call": round(dollars / max(paid, 1), 4)}
    print(f"\ncost: ${dollars:.2f} over {paid} paid calls")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"written {args.out}")
    return 0


def cmd_print_prompt(args) -> int:
    from isolation.pass_outputs import Call
    from isolation.program_runner import CallPaths

    made_up = [{"date": "2099-01-01", "risk_1yr": 20, "risk_2yr": 35,
                "reasons": [{"fact": "made-up fact", "file": "notes.txt", "date": "2098-12-30"}]}]
    call = Call("p000000000000-" + args.point, {"record": "r000000000000", "point": args.point, "arm": args.arm,
                                                "memory": made_up if args.arm == "memory" else []})
    print(program.prompt_text(call, CallPaths(program="/workspace/program", call="/workspace/call", ro={}), "PASS-ID"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pins")
    p.add_argument("--write", action="store_true")
    lc = sub.add_parser("leak-check")
    lc.add_argument("--split", choices=("val", "train", "all"), default="val")
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--dry-run", action="store_true", help="fake sessions: no Docker, no Claude, $0")
    r.add_argument("--chunk", type=int, default=c.CHUNK)
    s = sub.add_parser("score")
    s.add_argument("--run-id", required=True)
    s.add_argument("--out", type=Path)
    s.add_argument("--n-boot", type=int, default=sc.N_BOOT)
    pp = sub.add_parser("print-prompt")
    pp.add_argument("--point", default="m6", choices=c.POINT_KEYS)
    pp.add_argument("--arm", default="memory", choices=program.ARMS)
    args = ap.parse_args(argv)
    return {"pins": cmd_pins, "leak-check": cmd_leak_check, "run": cmd_run, "score": cmd_score,
            "print-prompt": cmd_print_prompt}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
