#!/usr/bin/env python3
"""One command for the whole task. Run from the engine checkout (or set AGENTIC_LABEL_OPT):

    python templates/new_task/run.py init             make the work folder and tag the seed program-v0
    python templates/new_task/run.py dry-run          the real loop with a scripted optimizer, $0, no Docker,
                                                      in its own throwaway folder (<work folder>-dryrun)
    python templates/new_task/run.py print-prompt     show exactly what the optimizer receives
    python templates/new_task/run.py build-images     build the optimizer and worker images (Docker)
    python templates/new_task/run.py pins --write     fingerprint both sealed containers and commit the pins
    python templates/new_task/run.py check-token      one tiny Claude call on your token alone, before any run
    python templates/new_task/run.py run --run-id r1  a real, sealed optimization run
    python templates/new_task/run.py report --run-id r1
    python templates/new_task/run.py test --tag program-v3   score ONE chosen version on TEST, once
    python templates/new_task/run.py score-val --tag program-v3   score one version on all 811 VAL patients

Exit codes for ``run``/``dry-run``: 0 finished, 3 stopped (the reason is printed and in the run
summary), 2 paused, 1 anything else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402  (also puts the engine on sys.path)
import adapters as ad  # noqa: E402
import optimizer as opt  # noqa: E402
import optimizer_prompt as op  # noqa: E402

from engine.driver import DriverConfig, StartPoint, exit_code, run_driver  # noqa: E402
from engine.loop import AgentOutcome, LoopPause, LoopStop  # noqa: E402
from engine.loop_ops import LocalLoopOps  # noqa: E402
from engine.materialize import materialize  # noqa: E402
from engine.run_start import Carried  # noqa: E402
from engine.run_store import CheckedRunStore  # noqa: E402
from engine.versioning import commit_new_version  # noqa: E402

PIN_FILE = "manifest.json"
#: Engine features this task relies on; the run stops before spending if the engine lacks one.
CAPABILITIES = (
    "engine.driver:run_driver",
    "engine.driver:DriverConfig.program_mode",
    "engine.schedule:DataSchedule",
    "isolation.sealed_worker:open_sealed_worker",
    "isolation.contained_agent:ContainedOptimizerAgent",
)
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


# ==================================================================================================
# init: the work folder and program-v0
# ==================================================================================================


def cmd_init(args) -> int:
    layout = opt.Layout()
    if layout.repo.exists():
        print(f"{layout.repo} already exists; delete {layout.root} to start over")
        return 1
    init_layout(layout)
    return 0


def init_layout(layout) -> None:
    for d in (layout.versions, layout.feedback, layout.notes, layout.outputs, layout.batches, layout.store,
              layout.archive, layout.transcripts, layout.stubs / "iter", layout.stubs / "feedback", layout.stubs / "notes"):
        d.mkdir(parents=True, exist_ok=True)
    (layout.private / "README.txt").write_text("Run records. Never mounted into a container.\n")
    (layout.stubs / "iter" / "README.txt").write_text("Releases the loop writes each iteration.\n")
    (layout.stubs / "notebook.md").write_text("# Notebook\n\n(empty: nothing tried yet)\n")
    shutil.copy2(layout.stubs / "notebook.md", layout.notebook)

    repo = layout.repo
    repo.mkdir(parents=True)
    git(repo, "init", "--initial-branch=main")
    git(repo, "config", "user.email", "optimizer@localhost")
    git(repo, "config", "user.name", f"{config.TASK_NAME} optimizer loop")
    shutil.copytree(config.SEED_PROGRAM_DIR, repo, dirs_exist_ok=True)
    (repo / "iter").mkdir()
    (repo / "iter" / "README.txt").write_text("Releases the loop writes each iteration.\n")
    (repo / ".gitignore").write_text("iter/\n__pycache__/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "seed program")
    tag, sha = commit_new_version(ad.TaskStore().manifest(), "seed", repo_root=repo, allow_empty=True)
    opt.stage_files(layout)
    print(f"work folder {layout.root}\nseed tagged {tag} at {sha[:10]}")


# ==================================================================================================
# The driver config, shared by dry-run and run
# ==================================================================================================


def schedule(layout):
    if config.SCHEDULE is None:
        return None
    from engine.schedule import DataSchedule, ScheduleConfig

    return DataSchedule(ScheduleConfig(**config.SCHEDULE), ad.TaskSource(layout.batches))


def driver_config(layout, *, run_id, iterations, agent, runner, release_builder, setup_owners, unsealed=(),
                  program_mode=None, resume=False, start=StartPoint.NEWEST_GOOD) -> DriverConfig:
    store = ad.TaskStore()
    sched = schedule(layout)
    layout.summary(run_id).parent.mkdir(parents=True, exist_ok=True)
    return DriverConfig(
        run_id=run_id, repo_root=layout.repo, iterations=iterations,
        program_store=store, scorer=ad.TaskScorer(), release_builder=release_builder, agent=agent, runner=runner,
        train_inputs=None if sched else ad.whole_split("train"),
        val_inputs=None if sched else ad.whole_split("val"),
        schedule=sched,
        materialize_root=layout.versions, manifest_for_version=lambda sha: store.manifest(),
        carried=(
            Carried(layout.repo / "iter", "run", stub=layout.stubs / "iter"),
            Carried(layout.feedback, "run", stub=layout.stubs / "feedback"),
            Carried(layout.notebook, "lineage", stub=layout.stubs / "notebook.md"),
            Carried(layout.notes, "lineage", stub=layout.stubs / "notes"),
        ),
        archive_root=layout.archive, run_summary_path=layout.summary(run_id),
        run_store=CheckedRunStore(layout.store), patient_data=config.PATIENT_DATA,
        engine_pin=config.ENGINE_PIN, engine_root=config.ENGINE_ROOT, engine_capabilities=CAPABILITIES,
        setup_owners=setup_owners, unsealed=unsealed, program_mode=program_mode,
        pin_command=f"python {HERE / 'run.py'} pins", start=start, resume=resume,
        loop_options={
            # Without loop_ops the loop writes no release files and the optimizer works blind.
            "loop_ops": LocalLoopOps(layout.repo),
            "build_mistake_corpus": ad.mistake_corpus(layout.feedback),
            "metric_field": config.METRIC_NAME,
            "hard_fields": {"task": config.TASK_NAME, "metric": config.METRIC_NAME, "schedule": config.SCHEDULE},
            "soft_fields": {"optimizer_model": config.OPTIMIZER_MODEL},
            "step_back_revert_tolerance": config.STEP_BACK_REVERT_TOLERANCE,
            "token_budget": config.TOKEN_BUDGET,
        },
    )


def drive(cfg: DriverConfig) -> int:
    try:
        result = run_driver(cfg)
    except (LoopStop, LoopPause) as exc:
        print(f"\nSTOPPED ({getattr(exc, 'reason', None) or type(exc).__name__}): {exc}")
        print(f"run summary: {cfg.run_summary_path}")
        return exit_code(exc)
    print(f"\nFINISHED {len(result.results)} iterations. best {result.best_tag} = {result.best_metric_value}")
    print(f"run summary: {cfg.run_summary_path}")
    return exit_code(result)


# ==================================================================================================
# dry-run: every piece but the two containers, for $0
# ==================================================================================================

# TASK-SPECIFIC: canned edits to the toy's label.py. Replace them with one or two hand-written improvements
# to your own seed, so the dry run shows the score move; or use {} (the scripted edit then appends a comment).
SCRIPTED_EDITS: dict = {}  # agentic: the scripted edit appends a comment to instructions.md


def stub_answer(pid: str) -> dict:
    """The dry run's stand-in answer for one patient: right at both horizons when the id ends in an even hex digit,
    "no" at both otherwise, and an unreadable answer when it ends in "f" (scored wrong). So a pass's balanced
    accuracy is a number scripts/check_task.py works out independently from the gold (a constant stand-in would
    score 0.5 whatever the scorer did)."""
    if pid[-1] == "f":
        return {"status": "ok", "id": pid, "error": "not_yes_or_no", "reason": "dry-run stand-in"}
    gold = next(r for r in ad.read_split("val") + ad.read_split("train") if r["id"] == pid)
    even = int(pid[-1], 16) % 2 == 0
    return {"status": "ok", "id": pid, "reason": "dry-run stand-in",
            **{key: (bool(gold[field]) if even else False) for key, field, _w in config.HORIZONS}}


class StubAgenticRunner(ad._Runner):
    """The dry run's stand-in for agentic sessions (which cost money): ``stub_answer`` per patient. Dry run only:
    it runs on the host, unsealed, and never reaches an optimizer."""

    class _Stub:
        def request(self, payload):
            return stub_answer(payload["id"])

        def close(self):
            pass

    def _open(self, tree):
        return self._Stub()


class ScriptedAgent:
    """Stands in for the optimizer in a dry run: writes a canned edit, and checks on the way that the
    loop really did hand it this iteration's feedback (the release files and the mistakes file)."""

    def __init__(self, layout) -> None:
        self.layout = layout
        self.checks: list[tuple[str, bool]] = []

    def run(self, *, iter_n: int, materialized_path=None) -> AgentOutcome:
        release = self.layout.repo / "iter" / str(iter_n) / "release_train.json"
        ok = release.is_file()
        mistakes = Path(json.loads(release.read_text())["corpus"]["mistakes_file"]) if ok else None
        self.checks.append((f"iter {iter_n}: release_train.json written", ok))
        self.checks.append((f"iter {iter_n}: mistakes file exists", bool(mistakes and mistakes.is_file())))
        edit = None if config.AGENTIC else SCRIPTED_EDITS.get(iter_n, None)
        program = self.layout.repo / config.EDITABLE_FILES[0]
        program.write_text(edit if edit else program.read_text() + f"\n# scripted edit {iter_n}\n")
        return AgentOutcome(exit_code=0, detail=f"scripted edit {iter_n}")


DRY_RUN_MARKER = ".alo-dry-run"


def remove_tree(root: Path) -> None:
    """Delete a work folder. Materialized versions are read-only, so make everything writable first."""
    for p in [root, *root.rglob("*")]:
        if not p.is_symlink():
            p.chmod(p.stat().st_mode | 0o700 if p.is_dir() else p.stat().st_mode | 0o600)
    shutil.rmtree(root)


def dry_run_layout():
    """The dry run's own work folder, rebuilt from scratch every time. Never the real one: the dry run
    commits its scripted edits as versions, and a real run starts from the newest version by default."""
    return opt.Layout(root=config.WORKDIR.with_name(config.WORKDIR.name + "-dryrun"))


def cmd_dry_run(args) -> int:
    layout = dry_run_layout()
    marker = layout.root / DRY_RUN_MARKER
    if layout.root.exists():
        if not marker.is_file():
            print(f"{layout.root} exists but has no {DRY_RUN_MARKER}, so it is not a dry-run folder; not deleting it")
            return 1
        remove_tree(layout.root)
    layout.root.mkdir(parents=True)
    marker.write_text("A dry-run work folder: run.py dry-run deletes and rebuilds it.\n")
    init_layout(layout)
    op.ITERATIONS = args.iterations
    agent = ScriptedAgent(layout)
    cfg = driver_config(
        layout, run_id=args.run_id, iterations=args.iterations, agent=agent,
        runner=StubAgenticRunner(layout.outputs) if config.AGENTIC else ad.LocalRunner(layout.outputs),
        release_builder=ad.TaskReleaseBuilder(None),
        setup_owners={}, unsealed=("optimizer", "program"), start=StartPoint.SEED,
    )
    code = drive(cfg)
    for name, ok in agent.checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(ok for _n, ok in agent.checks):
        return 1
    return code


# ==================================================================================================
# print-prompt
# ==================================================================================================


def cmd_print_prompt(args) -> int:
    from isolation.contained_agent import INPUTS_PATH, NOTES_PATH, OUT_PATH, OptimizerPaths

    names_in = (op.TASK_BRIEF_NAME, op.NOTEBOOK_NAME)
    names_out = (op.CHANGE_NOTE_NAME, op.NOTEBOOK_NAME)
    paths = OptimizerPaths(inputs={n: f"{INPUTS_PATH}/{n}" for n in names_in},
                           outputs={n: f"{OUT_PATH}/{n}" for n in names_out}, notes=NOTES_PATH)
    print("=" * 30, "SYSTEM PROMPT (prompts/optimizer_instructions.md)", "=" * 30)
    print(op.system_prompt())
    print("=" * 30, f"ITERATION {args.iter} PROMPT (optimizer_prompt.build_prompt)", "=" * 30)
    print(op.build_prompt(args.iter, paths))
    return 0


# ==================================================================================================
# Images and pins
# ==================================================================================================


def cmd_build_images(args) -> int:
    from isolation.image_pin import image_ref

    builds = [(config.OPTIMIZER_IMAGE_TAG, "optimizer.Dockerfile", ["--build-arg", f"CLAUDE_CODE_VERSION={config.CLAUDE_CODE_VERSION}"])]
    if not config.AGENTIC:  # agentic sessions run on the optimizer's image (it has Claude Code)
        builds.append((config.WORKER_IMAGE_TAG, "worker.Dockerfile", []))
    for tag, dockerfile, extra in builds:
        print(f"building {tag} ...", flush=True)
        subprocess.run(["docker", "build", *extra, "-t", tag, "-f", str(HERE / "docker" / dockerfile), str(HERE / "docker")],
                       check=True)
        print(f"  {image_ref(tag)}")
    return 0


def images() -> tuple[str, str]:
    """(optimizer image, program image), each name:version@sha256:... The agentic program's sessions
    run on the optimizer's image."""
    from isolation.image_pin import image_ref

    optimizer = image_ref(config.OPTIMIZER_IMAGE_TAG)
    return optimizer, optimizer if config.AGENTIC else image_ref(config.WORKER_IMAGE_TAG)


def program_runner(layout, *, image, pin):
    """The sealed runner for the program: a worker per pass (rules_based) or a ProgramRunner (agentic)."""
    if config.AGENTIC:
        import agentic

        return agentic.program_runner(layout, image=image, pin=pin, forbidden=opt.forbidden(layout))
    return ad.SealedRunner(layout.outputs, spec_for=opt.worker_spec_for(layout, image=image, pin=pin),
                           stderr_dir=layout.transcripts / "worker", versions_root=layout.versions)


def sealed_parts(layout, *, run_id, pins, token):
    """(optimizer, program runner, owners): ``owners`` maps each sealed kind to what the setup check fingerprints,
    the search sessions included when the proxy is on (a proxied optimizer without that owner is refused)."""
    optimizer_image, program_image = images()
    agent = opt.build_agent(layout, ad.TaskStore(), run_id=run_id, image=optimizer_image, pin=pins["optimizer"],
                            search_pin=pins.get("search"), token=token)
    runner = program_runner(layout, image=program_image, pin=pins["program"])
    owners = {"optimizer": agent, "program": runner}
    if config.SEARCH_PROXY:
        owners["search"] = agent.proxy
    return agent, runner, owners


#: The sealed kinds this task pins: the optimizer, the program, and the search sessions when the proxy is on.
PIN_KINDS = ("optimizer", "program", *(("search",) if config.SEARCH_PROXY else ()))


def cmd_pins(args) -> int:
    from isolation.setup_fingerprint import PIN_KEY, UNPINNED, fingerprint_hash, platform_key, setup_fingerprint

    layout = opt.Layout()
    _agent, _runner, owners = sealed_parts(layout, run_id="pins", pins=dict.fromkeys(PIN_KINDS, UNPINNED), token=None)
    prints = {kind: setup_fingerprint(owner.setup_description()) for kind, owner in owners.items()}
    hashes = {k: fingerprint_hash(v) for k, v in prints.items()}
    print(json.dumps(prints, indent=2))
    print(f"\nplatform {platform_key()}: {json.dumps(hashes, indent=2)}")
    if not args.write:
        print("\nreview the setup above, then rerun with --write to commit these pins")
        return 0
    path = layout.repo / PIN_FILE
    data = json.loads(path.read_text()) if path.exists() else {}
    pins = data.setdefault("runtime_pins", {}).setdefault(PIN_KEY, {})
    for kind, value in hashes.items():
        pins.setdefault(kind, {})[platform_key()] = value
    path.write_text(json.dumps(data, indent=2) + "\n")
    git(layout.repo, "add", PIN_FILE)
    git(layout.repo, "commit", "-m", f"setup pins for {platform_key()}")
    print(f"committed {path}")
    return 0


def committed_pins(layout) -> dict:
    from isolation.setup_fingerprint import committed_pin

    try:
        pins = {k: committed_pin(layout.repo, PIN_FILE, kind=k) for k in PIN_KINDS}
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"no committed setup pins ({exc}); run: python {HERE / 'run.py'} pins --write") from exc
    missing = [k for k, v in pins.items() if v is None]
    if missing:
        raise SystemExit(f"no committed setup pin for {missing}; run: python {HERE / 'run.py'} pins --write")
    return pins


# ==================================================================================================
# The Claude token
# ==================================================================================================

#: Auth variables a hand-run `claude` would prefer over the token; check-token removes them so it tests the
#: token alone. The engine passes only TOKEN_ENV_NAME into the containers, whatever else is set.
COMPETING_AUTH = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_BEDROCK")


def load_token() -> tuple[str | None, str]:
    """The optimizer's Claude token and where it came from: TOKEN_ENV_NAME, else TOKEN_FILE."""
    if os.environ.get(config.TOKEN_ENV_NAME):
        return os.environ[config.TOKEN_ENV_NAME].strip(), f"${config.TOKEN_ENV_NAME}"
    f = config.TOKEN_FILE
    if f.is_file():
        if f.stat().st_mode & 0o077:
            raise SystemExit(f"{f} can be read by other users: run chmod 600 {f}")
        token = f.read_text(encoding="utf-8").strip()
        # The agentic program's runner reads the token from this process's environment, by name, at call time
        # (isolation/program_runner.py); without this a file-only token reaches the optimizer but not the program.
        os.environ[config.TOKEN_ENV_NAME] = token
        return token, str(f)
    return None, ""


NO_TOKEN = (
    "No Claude token. The optimizer runs in a container that cannot see your Claude Code login, so it needs\n"
    "its own: run `claude setup-token` (it uses the subscription you are already signed into; pick your\n"
    "organization's account at the browser login) and save what it prints to {file} with mode 600,\n"
    "or export {env}. Then: python run.py check-token"
)


def cmd_check_token(args) -> int:
    """One tiny Haiku call that can only use the token: no login files (a throwaway HOME), and the auth
    variables that would win over it removed. Inside the optimizer image when it is built, so it also tests
    the route a run uses; else with the `claude` on this machine."""
    import tempfile

    token, source = load_token()
    if not token:
        print(NO_TOKEN.format(file=config.TOKEN_FILE, env=config.TOKEN_ENV_NAME))
        return 1
    for name in COMPETING_AUTH:
        if os.environ.get(name):
            print(f"note: {name} is set in this shell. Runs are unaffected (the engine passes only the token into\n"
                  f"      the containers), but a `claude` you run by hand here bills {name} instead of your seat.")
    env = {k: v for k, v in os.environ.items() if k not in COMPETING_AUTH}
    env[config.TOKEN_ENV_NAME] = token
    ask = ["claude", "-p", "Reply with the single word OK", "--model", "haiku", "--output-format", "json"]
    have_image = subprocess.run(["docker", "image", "inspect", config.OPTIMIZER_IMAGE_TAG],
                                capture_output=True).returncode == 0 if shutil.which("docker") else False
    with tempfile.TemporaryDirectory() as home:
        if have_image:
            where = f"the optimizer image ({config.OPTIMIZER_IMAGE_TAG})"
            cmd = ["docker", "run", "--rm", "-i", "-e", config.TOKEN_ENV_NAME, config.OPTIMIZER_IMAGE_TAG, *ask]
        elif shutil.which("claude"):
            where, cmd = "this machine's claude (build the images to test the run's own route)", ask
            env["HOME"] = home
        else:
            print("neither the optimizer image nor `claude` is available: run build-images first")
            return 1
        proc = subprocess.run(cmd, env=env, cwd=home, capture_output=True, text=True, timeout=180,
                              stdin=subprocess.DEVNULL)
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        out = {}
    ok = proc.returncode == 0 and out.get("is_error") is False
    print(f"token from {source}, tested in {where}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print((out.get("result") or proc.stderr or proc.stdout)[-600:])
        return 1
    print("The optimizer can authenticate with this token alone. Its usage counts against that account's seat.")
    return 0


# ==================================================================================================
# run: the real thing
# ==================================================================================================


def cmd_run(args) -> int:
    from isolation.program_mode import ProgramMode
    from isolation.setup_gate import check_setup

    layout = opt.Layout()
    op.ITERATIONS = args.iterations
    token, _source = load_token()
    if not config.ROUTE.is_vertex and not token:
        print(NO_TOKEN.format(file=config.TOKEN_FILE, env=config.TOKEN_ENV_NAME))
        return 1
    opt.stage_files(layout)
    agent, runner, owners = sealed_parts(layout, run_id=args.run_id, pins=committed_pins(layout),
                                         token=None if config.ROUTE.is_vertex else token)
    try:
        stamp = check_setup(owners, program_mode=ProgramMode(config.PROGRAM_MODE), runner=runner).value
    except LoopStop as exc:
        print(f"STOPPED ({exc.reason}): {exc}")
        return exit_code(exc)
    cfg = driver_config(
        layout, run_id=args.run_id, iterations=args.iterations, agent=opt.ChangeNoteAgent(agent), runner=runner,
        release_builder=ad.TaskReleaseBuilder(stamp), setup_owners=owners, program_mode=ProgramMode(config.PROGRAM_MODE),
        resume=args.resume, start=args.start,
    )
    return drive(cfg)


# ==================================================================================================
# report and test
# ==================================================================================================


def cmd_report(args) -> int:
    path = opt.Layout().summary(args.run_id)
    data = json.loads(path.read_text())
    metric = config.METRIC_NAME
    print(f"run {data.get('run_id')}  started from {data.get('start', {}).get('tag')}  stop: {data.get('stop')}")
    print(f"{'iter':>4}  {'version':<14}{'status':<18}{metric}")
    for row in data.get("iters", []):
        value = row.get(metric)
        print(f"{row.get('iter'):>4}  {row.get('tag', ''):<14}{row.get('status', ''):<18}"
              f"{'' if value is None else f'{value:.4f}'}{'   step-back' if row.get('is_step_back') else ''}")
    return 0


def cmd_test(args) -> int:
    """Score one chosen version on TEST. Every opening is recorded; a second look at TEST for the same
    version needs --again and is recorded as such, because each look spends TEST's credibility."""
    layout = opt.Layout()
    ledger = [json.loads(line) for line in layout.test_ledger.read_text().splitlines()] if layout.test_ledger.exists() else []
    earlier = [e for e in ledger if e["tag"] == args.tag]
    if earlier and not args.again:
        print(f"{args.tag} was already scored on TEST {len(earlier)} time(s): {earlier[-1]}. Pass --again to look again.")
        return 1
    sha = git(layout.repo, "rev-parse", f"{args.tag}^{{commit}}").strip()
    tree = layout.versions / f"test-{args.tag}"  # where program trees live; private/ is forbidden to the worker
    materialize(ad.TaskStore().manifest(), sha, repo_root=layout.repo, dest=tree)
    if args.local:
        if config.AGENTIC:
            print("--local runs a rules_based program in this process; an agentic program needs its sessions")
            return 1
        artifacts = ad.LocalRunner(layout.outputs).run(tree, ad.whole_split("test"))
    else:
        opt.stage_files(layout)
        runner = program_runner(layout, image=images()[1], pin=committed_pins(layout)["program"])
        if config.AGENTIC:
            with runner:  # a ProgramRunner: one network for the pass, containers removed on exit
                artifacts = runner.as_runner(run_id=f"test-{args.tag}-look{len(earlier) + 1}").run(tree, ad.whole_split("test"))
        else:
            artifacts = runner.run(tree, ad.whole_split("test"))
    if artifacts.status != "ok":
        print(f"TEST pass failed: {artifacts.status} {artifacts.error}")
        return 1
    score = ad.TaskScorer().score(artifacts, "test", {})
    entry = {"utc": datetime.now(timezone.utc).isoformat(), "tag": args.tag, "version_sha": sha,
             config.METRIC_NAME: score.primary_metric.value, "n": score.breakdown["n"],
             "look": len(earlier) + 1, "sealed": not args.local}
    with layout.test_ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=2))
    return 0


def cmd_score_val(args) -> int:
    """Score one chosen version on all of VAL (811 patients), not the 200-patient pool the rounds use: the tighter
    number to report for the version a run picked. VAL is not sealed, but every look is paid, so each is recorded.
    ``--pool`` scores the 200-patient pool instead, to compare an input arm (WITH_SUMMARY=1) with a run's own VAL
    number for the same version on the same patients (Phil 2026-10-29)."""
    layout = opt.Layout()
    token, _source = load_token()  # also puts a file-only token where the sessions' runner reads it
    if not config.ROUTE.is_vertex and not token:
        print(NO_TOKEN.format(file=config.TOKEN_FILE, env=config.TOKEN_ENV_NAME))
        return 1
    ledger_path = layout.private / "val_all_ledger.jsonl"
    arm = "summary" if config.WITH_SUMMARY else "record"
    scope = "pool200" if args.pool else "all"
    earlier = ([e for e in map(json.loads, ledger_path.read_text().splitlines())
                if e["tag"] == args.tag and e.get("arm", "record") == arm and e.get("scope", "all") == scope]
               if ledger_path.exists() else [])
    sha = git(layout.repo, "rev-parse", f"{args.tag}^{{commit}}").strip()
    name = f"val-{scope}-{arm}-{args.tag}" if (args.pool or config.WITH_SUMMARY) else f"val-all-{args.tag}"
    tree = layout.versions / name
    materialize(ad.TaskStore().manifest(), sha, repo_root=layout.repo, dest=tree)
    opt.stage_files(layout)
    runner = program_runner(layout, image=images()[1], pin=committed_pins(layout)["program"])
    with runner:  # a ProgramRunner: one network for the pass, containers removed on exit
        inputs = (ad.TaskSource(layout.batches).stage(sorted(ad.pools()["val"]), "val", f"{name}-ids")
                  if args.pool else ad.whole_split("val"))
        artifacts = runner.as_runner(run_id=f"{name}-look{len(earlier) + 1}").run(tree, inputs)
    if artifacts.status != "ok":
        print(f"VAL pass failed: {artifacts.status} {artifacts.error}")
        return 1
    score = ad.TaskScorer().score(artifacts, "val", {})
    entry = {"utc": datetime.now(timezone.utc).isoformat(), "tag": args.tag, "version_sha": sha,
             config.METRIC_NAME: score.primary_metric.value, "n": score.breakdown["n"], "look": len(earlier) + 1,
             "arm": arm, "scope": scope}
    with ledger_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=2))
    return 0


# ==================================================================================================


def run_id_arg(value: str) -> str:
    if not RUN_ID.match(value):
        raise argparse.ArgumentTypeError("a run id is letters, digits, '.', '_' or '-', at most 64")
    return value


def start_arg(value: str):
    return {"newest": StartPoint.NEWEST_GOOD, "seed": StartPoint.SEED}.get(value, value)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    d = sub.add_parser("dry-run")
    d.add_argument("--run-id", type=run_id_arg, default="dry-run")
    d.add_argument("--iterations", type=int, default=2)
    pp = sub.add_parser("print-prompt")
    pp.add_argument("--iter", type=int, default=2)
    sub.add_parser("build-images")
    sub.add_parser("check-token")
    pins = sub.add_parser("pins")
    pins.add_argument("--write", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--run-id", type=run_id_arg, required=True)
    r.add_argument("--iterations", type=int, default=config.ITERATIONS)
    r.add_argument("--start", type=start_arg, default=StartPoint.NEWEST_GOOD, help="newest | seed | a tag")
    r.add_argument("--resume", action="store_true", help="continue this run id where it stopped")
    rep = sub.add_parser("report")
    rep.add_argument("--run-id", type=run_id_arg, required=True)
    t = sub.add_parser("test")
    t.add_argument("--tag", required=True)
    t.add_argument("--again", action="store_true")
    t.add_argument("--local", action="store_true", help="run unsealed in this process (only for checking the template)")
    sv = sub.add_parser("score-val", help="score one chosen version on all of VAL, not the 200-patient pool")
    sv.add_argument("--tag", required=True)
    sv.add_argument("--pool", action="store_true", help="the 200-patient VAL pool the rounds use, not all of VAL")
    args = p.parse_args(argv)
    return {
        "init": cmd_init, "dry-run": cmd_dry_run, "print-prompt": cmd_print_prompt, "build-images": cmd_build_images,
        "pins": cmd_pins, "check-token": cmd_check_token, "run": cmd_run, "report": cmd_report, "test": cmd_test,
        "score-val": cmd_score_val,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
