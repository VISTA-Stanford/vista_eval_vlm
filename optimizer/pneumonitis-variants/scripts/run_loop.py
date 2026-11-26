"""The 5-round builder loop on TRAIN (spec planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md, Decided:
"Builder (Opus) -> 5-round loop on TRAIN: each round the reader answers fresh TRAIN scans and the builder sees its
mistakes with scans and reports, then revises guide and examples"). Adapted from ct-findings-5's knowledge-base loop
(scripts/run_kb_loop.py): an Opus builder with the engine's search proxy in place of the Sonnet builder, the guide and
the example library in place of the KB library, and the reader's mistakes (with their CT and report) fed back.

    init                       the work folder (run.py init + pins), the builder's scans folder with the source scans
    pins [--write]             fingerprint the builder session; with --write commit its pin beside run.py's
    loop [--run-id loop-r1] [--cap 250] [--already-spent X] [--val-scans 134]        PAID
    report [--run-id loop-r1] [--out F]   $0: each round's TRAIN mean, per-variant AUROC, library size, spend
    print-prompt [--iter 2]    $0: what the builder receives

Round k (1..5):
  1. builder  a sealed Opus session (the engine's contained editing session, through the search proxy) edits guide/
              and examples/ of program-v<k-1>. It reads scans/ (the 60 source scans, plus every earlier round's
              mistakes) and reader/round-<j>/ for j < k. Its version must pass tools/examples.py check, every entry
              must come from a scan in scans/ with that entry's variant as its answer, and the guide must not be the
              empty placeholder; else the round stops. Committed as program-v<k>.
  2. reader   one sealed Sonnet session per scan (the program runner, as run.py val-pass) on round k's 80 TRAIN scans
              (rounds.json) with program-v<k>; scored per variant (AUROC) on those scans: the round's TRAIN mean.
  3. feedback reader/round-<k>/ for the next builder (per scan: answer, probabilities, the cause ranked first, reasons,
              findings note, drawn images opened); each mistake (the cause ranked first is not the answer, or the
              session failed) is added to scans/ with its CT, organ ranges, answer and report.
After round 1 the loop stops if fewer than half the reader's sessions opened every drawn image (an early form of the
spec's guide-arm stop rule). Stops before a step if the spend could pass --cap with the guide arm's VAL pass kept back.
Resumes: a finished step of a round (private/loop/<run>/state.json) is not redone. Prints counts, scores, dollars only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import adapters as ad  # noqa: E402
import agentic  # noqa: E402
import optimizer as opt  # noqa: E402
import optimizer_prompt as op  # noqa: E402
import run  # noqa: E402
from engine.materialize import materialize  # noqa: E402
from engine.versioning import commit_new_version  # noqa: E402

sys.path.insert(0, str(HERE / "scripts"))
from usage import usage  # noqa: E402

BUILDER_BUDGET_USD = config.OPTIMIZER_SESSION_BUDGET_USD
PIN_KIND = "builder"
RETIRED = "retired.json"
_GUIDE_SEED = config.SEED_PROGRAM_DIR / "guide" / "README.md"  # the library program has no guide/
EMPTY_GUIDE = _GUIDE_SEED.read_text(encoding="utf-8") if _GUIDE_SEED.is_file() else ""
VARIANTS = tuple(k for k, _t, _q in config.QUESTIONS)


def layout():
    return opt.Layout()


def scans_dir(lay) -> Path:
    return lay.root / "builder-scans"


def reader_dir(lay) -> Path:
    return lay.root / "reader-feedback"


def state_path(lay, run_id: str) -> Path:
    return lay.private / "loop" / run_id / "state.json"


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


def gold_variant(row: dict) -> str:
    return next(v for v in VARIANTS if row.get(v) == 1)


# --------------------------------------------------------------------------------------------------
# The builder's scans
# --------------------------------------------------------------------------------------------------


def add_scans(lay, ids: list[str], origin: str) -> int:
    """Copy TRAIN scans into scans/<id>/ (ct/, organs.json, answer.json, report.txt) and list them in index.json.
    TRAIN only: a VAL id raises. Every file is read-only. Returns how many were new."""
    train = {r["id"]: r for r in ad.read_split("train")}
    root = scans_dir(lay)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else []
    have = {e["scan"] for e in index}
    new = 0
    for i in ids:
        if i not in train:
            raise SystemExit("a scan handed to the builder is not a TRAIN pool scan")
        if i in have:
            continue
        d = root / i
        (d / "ct").mkdir(parents=True, exist_ok=True)
        for name in ("ct.npz", "meta.json"):
            shutil.copy2(ad.patient_dir(i) / name, d / "ct" / name)
        shutil.copy2(config.DATA_DIR / "organs" / f"{i}.json", d / "organs.json")
        (d / "answer.json").write_text(json.dumps({"cause": gold_variant(train[i])}))
        report = ad.train_report(i)
        if report is not None:
            shutil.copy2(report, d / "report.txt")
        for f in d.rglob("*"):
            if f.is_file():
                f.chmod(0o444)
        index.append({"scan": i, "cause": gold_variant(train[i]), "from": origin})
        have.add(i)
        new += 1
    index_path.chmod(0o644) if index_path.exists() else None
    index_path.write_text(json.dumps(index, indent=1))
    index_path.chmod(0o444)
    (root / "README.md").write_text(
        "# Your labelled TRAIN scans\n\nOne folder per scan: ct/ (ct.npz, meta.json: the same files the reader gets), "
        "organs.json, answer.json (the cause: the answer key), report.txt (the radiology report the answer was read "
        "from). index.json lists every scan with its cause and where it came from: 'source' (given before round 1) or "
        "'round-<k>' (a scan the reader got wrong in round k; its feedback is in the reader folder). Draw entries with "
        "`python3 -B tools/examples.py render <scans>/<id>/ct --organs <scans>/<id>/organs.json --view '...' --out ...`.\n")
    return new


def stage_brief(lay) -> None:
    opt.stage_files(lay)  # worker.py and the brief (context/task.md -> inputs/task.md)
    if config.LIBRARY:  # the literature review: the mount is forbidden to every session, so a copy
        shutil.copy2(config.LITERATURE_V2, lay.inputs / "literature.md")


# --------------------------------------------------------------------------------------------------
# The builder session
# --------------------------------------------------------------------------------------------------


def atlas_prompt(iter_n: int, paths) -> str:
    """The atlas builder (config.ATLAS): session 1 is the literature review (round 0); session k+1 is atlas round k."""
    reader, k = paths.inputs["reader"], iter_n - 1
    if k == 0:
        first = ("This is round 0: the guide from the literature. guide/literature.md is a review of the published "
                 "literature on how the five causes are described and seen on CT, with its sources (each marked full "
                 "text or abstract). Read it closely. Write a first guide/README.md: a `## Patterns` section built from "
                 "the review, each pattern citing it, then how to weigh the patterns into five probabilities. Read some "
                 "source scans' reports to learn how this dataset's radiologists word each cause. You may correct or "
                 "tighten literature.md (keep its format and sources), but you cannot search the web here. Do not "
                 "build atlas entries yet. `python3 -B tools/atlas.py check` must PASS on the literature.")
    elif k == 1:
        first = ("This is atlas round 1: the guide and literature review exist (read them first); the atlas is empty. "
                 f"Build the atlas: exactly {config.ATLAS_PER_VARIANT} entries per cause from your scans, each written "
                 "up from its report, its answer and its images, findings tied to slice numbers, and marked. Then "
                 "revise the guide so each pattern points to the atlas entries and slices that show it.")
    else:
        first = (f"This is atlas round {k}. The reader has read rounds 1 to {k - 1} with your earlier versions; its "
                 f"feedback is in `{reader}/round-<j>/`, and the scans it got wrong are now in your scans folder. Start "
                 "with the latest round: what it got wrong and why. Then revise the guide, the literature review and "
                 f"the atlas (swap entries to keep exactly {config.ATLAS_PER_VARIANT} per cause).")
    return f"""{first}

## Where things are

- `{paths.inputs['task.md']}`: the task brief. Read it first.
- `{paths.inputs['scans']}/`: your labelled TRAIN scans (README.md and index.json there).
- `{reader}/`: the reader's feedback, one folder per finished round (empty until round 1 is read).
- `guide/` and `atlas/` in your working folder (`/workspace/rw/program`): edit files under these two only;
  `instructions.md` and `tools/` are the reader's and must not change (a change there refuses your whole edit).
- `tools/atlas.py` (draw, list, check), `tools/view_ct.py`, `tools/zoom_organ.py`: the reader's own tools.
- `{paths.inputs['notebook.md']}`: your notebook so far. Copy it to `{paths.outputs['notebook.md']}` first, then
  edit it there.
- `{paths.outputs['change_note.json']}`: write this round's change note here (a JSON object: what you changed and why).
- `{paths.versions}/`: earlier versions of the program, as plain folders.

Images: every image you open stays in this session's memory, and the session fails once they add up past about 30 MB,
losing everything you built. Explore scans at `--size 768` or less, draw each atlas image without marks to /tmp at
`--size 640`, look at it once, place the marks, then write the entry's image with `atlas.py draw` (768 px) without
opening it again.

Removing entries: you cannot delete files here (a deleted file refuses your whole edit). To remove an entry, add its
folder name to `atlas/{RETIRED}`, a JSON list of folder names; the loop deletes those entries after your session.

Before you finish, run `python3 -B tools/atlas.py check` from your working folder and fix every problem it prints.
"""


#: Discovery's seven sessions, in order (spec 2026-11-25-ct-pneumonitis-library.md); the climb's director follows.
DISCOVERY_ROLES = ("director-plan", *VARIANTS, "director-synth")
ROLE: dict = {"role": None, "round": None}


def library_prompt(iter_n: int, paths) -> str:
    """The library arm's builder prompt for the current ROLE (a discovery role, or the climb director)."""
    role, k = ROLE["role"], ROLE["round"]
    if role == "director-plan":
        first = ("You are the director, session 1 of discovery. Read the literature review closely and look through the "
                 "literature images' index. Plan the library: the patterns to look for for each cause (the literature's "
                 "terms), the hard pairs, and what each of the five managers should find and count in our 599 TRAIN "
                 "scans. Write a first start page (pages/index.md), one page per cause and per pattern from the literature "
                 "(what it says, how often, which literature images show it), and catalogue/plan.md with the managers' "
                 "instructions. Do not add Stanford examples yet; a few subagents reading reports to size the work is fine.")
    elif role in VARIANTS:
        first = (f"You are the manager for {role}, a discovery session. Read catalogue/plan.md and the pages so far. Find in "
                 f"our 599 TRAIN scans what the literature describes for {role}, and the variants it does not describe: fan "
                 "out subagents over the reports (Grep) and the images, count each pattern across the scans of every cause "
                 f"(the look-alikes matter as much), and add up to 30 Stanford examples of {role} (clear ones, and ones that "
                 f"show a look-alike), each written up and marked. Update the pattern and cause pages with what our data "
                 f"shows, and write catalogue/{role}.json (pattern -> counts by cause, and how you counted). Leave the other "
                 "causes' examples and catalogues as they are.")
    elif role == "director-synth":
        first = ("You are the director, the last discovery session. Reconcile the five managers' work into one library: "
                 "merge or split patterns, resolve contradictions, write catalogue/counts.json and catalogue/README.md "
                 "(pattern x cause from all five catalogues), write the hard-pair pages (what the image can and cannot "
                 "separate, from the literature and our counts; when to stay near even odds), rewrite the start page as a "
                 "short reading procedure, and keep at most 150 Stanford examples (retire the weakest). Note in the notebook "
                 "the literature you would ask for in the next run.")
    else:
        first = (f"You are the director, climb round {k}. The reader has read rounds 1 to {k - 1} with the library; its "
                 f"feedback is in `{paths.inputs['reader']}/round-<j>/` (answers, findings notes, how it used the library). "
                 "Start with the latest round: what it got wrong, which pages and examples it used and whether they "
                 "helped. Revise the library: pages, examples (swap within the 150), the catalogue, the start page. Fan "
                 "out subagents as you need. Note literature you would ask for in the next run in the notebook.")
    return f"""{first}

## Where things are
- `{paths.inputs['task.md']}`: the task brief. Read it first.
- `{paths.inputs['literature.md']}`: the literature review. `{paths.inputs['lit']}/`: the literature images and index.json.
- `{paths.inputs['train']}/`: the 599 labelled TRAIN scans (README.md, index.json, _tools/similar.py).
- `{paths.inputs['reader']}/`: the reader's feedback, one folder per finished climb round (empty during discovery).
- `library/` in your working folder (`/workspace/rw/program`): edit files under it only; `instructions.md` and `tools/`
  are the reader's and must not change (a change there refuses your whole edit). tools/library.py (the reader's tool;
  `check` for you), tools/atlas.py (draw), tools/view_ct.py, tools/zoom_organ.py.
- `{paths.inputs['notebook.md']}`: the team's notebook so far. Copy it to `{paths.outputs['notebook.md']}` first, then
  add to it there.
- `{paths.outputs['change_note.json']}`: this session's change note (a JSON object: what you did and why).
- `{paths.versions}/`: earlier versions of the program, as plain folders.
"""


def builder_prompt(iter_n: int, paths) -> str:
    if config.LIBRARY:
        return library_prompt(iter_n, paths)
    if config.ATLAS:
        return atlas_prompt(iter_n, paths)
    reader = paths.inputs["reader"]
    first = ("This is round 1: the guide is empty, the library has no entries, and there is no reader feedback yet. "
             "Build a first guide and a first library from the reference literature and the source scans."
             if iter_n == 1 else
             f"This is round {iter_n}. The reader has read rounds 1 to {iter_n - 1} with your earlier versions; its "
             f"feedback is in `{reader}/round-<k>/`, and the scans it got wrong are now in your scans folder. Start "
             "with the latest round: what it got wrong and why. Then revise the guide and the library.")
    return f"""{first}

## Where things are

- `{paths.inputs['task.md']}`: the task brief. Read it first.
- `{paths.inputs['scans']}/`: your labelled TRAIN scans (README.md and index.json there).
- `{reader}/`: the reader's feedback, one folder per finished round (empty in round 1).
- `guide/` and `examples/` in your working folder (`/workspace/rw/program`): edit files under these two only;
  `instructions.md` and `tools/` are the reader's and must not change (a change there refuses your whole edit).
- `tools/examples.py` and `tools/view_ct.py`, `tools/zoom_organ.py`: the reader's own tools. Draw entry images with
  `python3 -B tools/examples.py render`.
- `{paths.inputs['notebook.md']}`: your notebook so far. Copy it to `{paths.outputs['notebook.md']}` first, then
  edit it there.
- `{paths.outputs['change_note.json']}`: write this round's change note here (a JSON object: what you changed and why).
- `{paths.versions}/`: earlier versions of the program, as plain folders.

Images: every image you open stays in this session's memory, and the session fails once they add up past about 30 MB,
losing everything you built. So explore scans at `--size 768` or less (view_ct.py and examples.py render both take
--size), check each entry image right after you render it with a preview at `--size 640` written to /tmp (then write
the entry's own image at the default size without opening it again), open each image once, and add at most 20 new
entries this round.

Removing entries: you cannot delete files here (a deleted file refuses your whole edit). To remove an entry, add its
folder name to `examples/{RETIRED}`, a JSON list of folder names; the loop deletes those entries after your session.
To change an entry, edit its files in place.

Before you finish, run `python3 -B tools/examples.py check` from your working folder and fix every problem it prints.
"""


def build_builder(lay, *, run_id: str, pin: str, search_pin: str):
    from engine.search_proxy import ProxiedOptimizer
    from isolation.contained_agent import ContainedOptimizerAgent, DeclaredOutput

    prof, credential = config.ROUTE.session_auth()
    token = None if config.ROUTE.is_vertex else run.load_token()[0]
    reader_dir(lay).mkdir(parents=True, exist_ok=True)
    proxy = opt.search_proxy(lay, image=run.images()[0], pin=search_pin, token=token)  # None when SEARCH_PROXY is off
    agent = ContainedOptimizerAgent(
        model=config.OPTIMIZER_MODEL,
        system_prompt=op.system_prompt(),
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
        readable_inputs=(((op.TASK_BRIEF_NAME, lay.inputs / op.TASK_BRIEF_NAME), (op.NOTEBOOK_NAME, lay.notebook),
                          ("literature.md", lay.inputs / "literature.md"), ("lit", config.LIT_ASSETS_DIR / "lit"),
                          ("train", config.TRAIN_SCANS_DIR), ("reader", reader_dir(lay))) if config.LIBRARY else
                         ((op.TASK_BRIEF_NAME, lay.inputs / op.TASK_BRIEF_NAME), (op.NOTEBOOK_NAME, lay.notebook),
                          ("scans", scans_dir(lay)), ("reader", reader_dir(lay)))),
        declared_outputs=(DeclaredOutput(op.CHANGE_NOTE_NAME), DeclaredOutput(op.NOTEBOOK_NAME, destination=lay.notebook)),
        notes_root=lay.notes,
        run_id=run_id,
        validation_roots=(lay.private, config.DATA_DIR),
        max_budget_usd=BUILDER_BUDGET_USD,
        effort=config.OPTIMIZER_EFFORT,
        web_search=False,
        search_grant=proxy.grant if proxy else None,
        timeout_seconds=config.OPTIMIZER_TIMEOUT_SECONDS,
        host_env={config.TOKEN_ENV_NAME: token} if token else None,
        env=opt.ENV,
        patient_data=config.PATIENT_DATA,
    )
    return ProxiedOptimizer(agent, proxy) if proxy else agent


def retire_entries(library: Path, sub: str = "entries") -> list[str]:
    """Delete the entries listed in examples/retired.json (the session may not delete files), then the list."""
    f = library / RETIRED
    if not f.is_file():
        return []
    try:
        names = json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        names = []
    f.unlink()
    removed = []
    for name in names if isinstance(names, list) else []:
        d = library / sub / str(name)
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", str(name)) and (d / "entry.json").is_file():
            shutil.rmtree(d)
            removed.append(str(name))
    return removed


def load_atlas_tool(tree: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"atlas_{abs(hash(str(tree)))}", Path(tree) / "tools" / "atlas.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(tree) / "tools"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(Path(tree) / "tools"))
        sys.modules.pop("examples", None)
        sys.modules.pop("view_ct", None)
    mod.PROGRAM = Path(tree)
    mod.ATLAS, mod.GUIDE = mod.PROGRAM / "atlas", mod.PROGRAM / "guide"
    mod.ENTRIES = mod.ATLAS / "entries"
    return mod


def atlas_version_problems(tree: Path, allowed: dict[str, str], literature_only: bool) -> list[str]:
    """The atlas tool's own checks (atlas and literature), plus: every entry from one of the builder's scans with that
    scan's answer as its variant; entries required from atlas round 1, none in round 0."""
    tool = load_atlas_tool(tree)
    bad = list(tool.literature_problems()) + list(tool.atlas_problems())
    rows = tool.entries()
    if literature_only and rows:
        bad.append("round 0 builds no atlas entries")
    if not literature_only and not rows:
        bad.append("the atlas has no entries")
    for eid, e in rows:
        src = e.get("source")
        if src not in allowed:
            bad.append(f"{eid}: source is not one of your scans")
        elif e.get("variant") != allowed[src]:
            bad.append(f"{eid}: variant is not its scan's answer")
    return bad


def load_library_tool(tree: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"library_{abs(hash(str(tree)))}", Path(tree) / "tools" / "library.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(tree) / "tools"))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(Path(tree) / "tools"))
        sys.modules.pop("examples", None)
        sys.modules.pop("view_ct", None)
    mod.PROGRAM = Path(tree)
    mod.LIBRARY = mod.PROGRAM / "library"
    mod.PAGES, mod.EXAMPLES, mod.CATALOGUE = mod.LIBRARY / "pages", mod.LIBRARY / "examples", mod.LIBRARY / "catalogue"
    mod._call = None
    return mod


def train_causes() -> dict[str, str]:
    return {r["id"]: r["cause"] for r in json.loads((config.TRAIN_SCANS_DIR / "index.json").read_text())}


def library_version_problems(tree: Path, *, examples_allowed: bool = True) -> list[str]:
    """The reader's library check (pages, links, examples, size), plus: every example from a train/ scan with that
    scan's answer as its cause; no examples in the director's planning session."""
    tool = load_library_tool(tree)
    lit_ids = {r["id"] for r in json.loads((config.LIT_ASSETS_DIR / "lit" / "index.json").read_text())}
    bad = list(tool.library_problems(lit_ids))
    allowed = train_causes()
    exs = tool.examples()
    if not examples_allowed and exs:
        bad.append("the planning session adds no Stanford examples")
    for eid, e in exs.items():
        src = e.get("source")
        if src not in allowed:
            bad.append(f"example {eid}: source is not one of the TRAIN scans")
        elif e.get("cause") != allowed[src]:
            bad.append(f"example {eid}: cause is not its scan's answer")
    return bad


def version_problems(tree: Path, allowed: dict[str, str]) -> list[str]:
    """The reader's own check over a version's library, plus: every entry's source is one of the builder's scans
    (``allowed``: id -> cause) and its variant is that scan's cause; the guide is written (not the empty placeholder)."""
    import stage_examples

    tool = stage_examples.load_tool(tree)
    bad = list(tool.library_problems())
    rows = tool.entries()
    if not rows:
        bad.append("the library has no entries")
    for eid, e in rows:
        src = e.get("source")
        if src not in allowed:
            bad.append(f"{eid}: source is not one of your scans")
        elif e.get("variant") != allowed[src]:
            bad.append(f"{eid}: variant is not its scan's answer")
    guide = tree / "guide" / "README.md"
    if not guide.is_file() or guide.read_text(encoding="utf-8").strip() == EMPTY_GUIDE.strip():
        bad.append("guide/README.md is missing or still the empty placeholder")
    size = sum(f.stat().st_size for f in (tree / "guide").rglob("*") if f.is_file())
    if size > 60_000:
        bad.append(f"the guide is {size} bytes; keep it under about 40 KB")
    return bad


def library_stats(tree: Path) -> dict:
    if config.LIBRARY:
        tool = load_library_tool(tree)
        exs = tool.examples()
        text = sum(f.stat().st_size for f in (tree / "library").rglob("*") if f.is_file() and f.suffix in (".md", ".json"))
        return {"pages": len(tool.pages()), "examples": len(exs), "by_cause": dict(Counter(e.get("cause") for e in exs.values())),
                "text_bytes": text}
    if config.ATLAS:
        tool = load_atlas_tool(tree)
        rows = [e for _eid, e in tool.entries()]
        guide = sum(f.stat().st_size for f in (tree / "guide").rglob("*") if f.is_file())
        return {"entries": len(rows), "by_variant": dict(Counter(r.get("variant") for r in rows)), "guide_bytes": guide,
                "sources": tool.cited_sources()}
    rows = [json.loads(f.read_text()) for f in sorted((tree / "examples" / "entries").glob("*/entry.json"))]
    guide = sum(f.stat().st_size for f in (tree / "guide").rglob("*") if f.is_file())
    return {"entries": len(rows), "by_variant": dict(Counter(r.get("variant") for r in rows)), "guide_bytes": guide}


# --------------------------------------------------------------------------------------------------
# The reader and its feedback
# --------------------------------------------------------------------------------------------------


def ranked_first(a: dict) -> str | None:
    if a.get("error"):
        return None
    return max(VARIANTS, key=lambda v: (ad.prob(a, v), -VARIANTS.index(v)))


def write_feedback(lay, k: int, artifacts, score, tree: Path) -> tuple[dict, list[str]]:
    """The reader's round k as the builder will read it; returns (summary, mistake ids)."""
    import stage_examples

    gold = {r["id"]: r for r in ad.read_split("train")}
    answers = ad._answers(artifacts)
    out = reader_dir(lay) / f"round-{k}"
    if out.exists():
        for p in out.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
        shutil.rmtree(out)
    (out / "findings").mkdir(parents=True)
    scans, mistakes = [], []
    for i, a in sorted(answers.items()):
        events = next((f for f in ad._trace_files(artifacts, i) if f.name == "events.jsonl"), None)
        u = usage(events)
        if config.LIBRARY:
            u.update(library_use(a.get("library_log")))
            staged = 0
        else:
            staged = len(stage_examples.planned_atlas(tree, i) if config.ATLAS else stage_examples.planned(tree, i))
        top, truth = ranked_first(a), gold_variant(gold[i])
        wrong = top != truth
        if wrong:
            mistakes.append(i)
        note = Path(a["findings_log"]) if a.get("findings_log") else None
        if note and note.is_file():
            shutil.copy2(note, out / "findings" / f"{i}.md")
        scans.append({"scan": i, "answer": truth, "ranked_first": top, "mistake": wrong, "error": a["error"],
                      "probability": {v: (None if a["error"] else round(100 * ad.prob(a, v))) for v in VARIANTS},
                      "reasons": a.get("reasons") or [],
                      "findings_note": f"findings/{i}.md" if (out / "findings" / f"{i}.md").is_file() else None,
                      "in_your_scans": wrong, "drawn_images_opened": u["drawn_opened"],
                      "drawn_images_staged": staged, "read_guide": u["read_guide"],
                      **({k2: u[k2] for k2 in ("library_commands", "pages_read", "examples_opened", "lit_images_opened",
                                               "searches")} if config.LIBRARY else {})})
    per_q = run.per_question(score)
    opened_all = [s for s in scans if s["drawn_images_staged"] and s["drawn_images_opened"] >= s["drawn_images_staged"]]
    summary = {
        "round": k, "scans": len(scans), "mean_auroc": round(score.primary_metric.value, 4),
        "auroc": {v: per_q[v]["auroc"] for v in VARIANTS},
        "ranked_first_right": sum(not s["mistake"] for s in scans),
        "confusions": dict(Counter(f"{s['answer']}->{s['ranked_first']}" for s in scans if s["mistake"])),
        "share_opened_all_drawn_images": round(len(opened_all) / max(len(scans), 1), 3),
        "share_read_guide": round(sum(s["read_guide"] for s in scans) / max(len(scans), 1), 3),
        **({"share_used_library": round(sum(s["library_commands"] > 0 for s in scans) / max(len(scans), 1), 3),
            "mean_library_commands": round(sum(s["library_commands"] for s in scans) / max(len(scans), 1), 2),
            "mean_examples_opened": round(sum(s["examples_opened"] for s in scans) / max(len(scans), 1), 2),
            "mean_lit_images_opened": round(sum(s["lit_images_opened"] for s in scans) / max(len(scans), 1), 2),
            "pages_read": dict(Counter(p for s in scans for p in s["pages_read"]).most_common(30))} if config.LIBRARY else {}),
        "failed_sessions": sum(bool(s["error"]) for s in scans),
    }
    (out / "scans.json").write_text(json.dumps(scans, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "README.md").write_text(
        f"# Reader, round {k}\n\nThe reader read {len(scans)} TRAIN scans with program-v{k}.\n\n"
        "- summary.json: mean and per-cause AUROC on these scans, how often the cause it ranked first was the answer, "
        "the confusions (answer->ranked first), and how much it used the guide and the drawn images.\n"
        "- scans.json: per scan, the answer, the reader's probabilities (0-100), the cause it ranked first, its stated "
        "reasons, and how many drawn images it opened. A mistake (ranked first is not the answer) is now in your scans "
        "folder with its CT and report.\n- findings/<scan>.md: the reader's own findings note.\n")
    for p in out.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    return summary, mistakes


def library_use(log_path) -> dict:
    """One reader session's library use, from its library_log.jsonl (tools/library.py writes one line per command)."""
    rows = []
    if log_path and Path(log_path).is_file():
        for x in Path(log_path).read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(x))
            except ValueError:
                continue
    return {"library_commands": len(rows),
            "pages_read": sorted({r.get("name") for r in rows if r.get("cmd") == "page" and r.get("found")}),
            "examples_opened": sum(r.get("cmd") == "example" and bool(r.get("found")) for r in rows),
            "lit_images_opened": sum(r.get("cmd") == "lit" and bool(r.get("found")) for r in rows),
            "searches": sum(r.get("cmd") == "search" for r in rows)}


def pass_cost(run_id: str) -> tuple[float, int]:
    """Dollars and sessions of every finished reader pass under this pass run id, from the runner's receipts."""
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
    return round(sum(float(r.get("builder_usd") or 0) + float(r.get("reader_usd") or 0)
                     for r in state["rounds"].values()), 4)


# --------------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------------


def cmd_init(_args) -> int:
    lay = layout()
    if not lay.repo.exists():
        run.init_layout(lay)
    stage_brief(lay)
    reader_dir(lay).mkdir(parents=True, exist_ok=True)
    if config.LIBRARY:  # the caches every session links from: all discovery TRAIN scans, the literature images
        import subprocess

        for step in ("train", "lit", "check"):
            subprocess.run([sys.executable, "-B", str(HERE / "scripts" / "library_assets.py"), step], check=True)
        if not (lay.repo / run.PIN_FILE).exists():
            run.cmd_pins(argparse.Namespace(write=True))
        print(f"work folder {lay.root}; caches {config.TRAIN_SCANS_DIR}, {config.LIT_ASSETS_DIR}")
        return 0
    rounds = json.loads((config.DATA_DIR / "rounds.json").read_text())
    new = add_scans(lay, rounds["source"], "source")
    if not (lay.repo / run.PIN_FILE).exists():
        run.cmd_pins(argparse.Namespace(write=True))
    print(f"work folder {lay.root}; builder scans {len(json.loads((scans_dir(lay) / 'index.json').read_text()))} "
          f"({new} new)")
    return 0


def cmd_pins(args) -> int:
    from isolation.setup_fingerprint import PIN_KEY, UNPINNED, fingerprint_hash, platform_key, setup_fingerprint

    lay = layout()
    stage_brief(lay)
    agent = build_builder(lay, run_id="pins", pin=UNPINNED, search_pin=UNPINNED)
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
        raise SystemExit("no committed builder pin: run scripts/run_loop.py pins --write")
    return pin


def commit_builder_session(lay, state, rec, *, base: str, iter_n: int, label: str, bpin: str, pins: dict,
                           cap_left) -> int:
    """One library builder session on ``base``: run it (one retry), retire listed examples, commit, check. Fills rec
    with tag, version_sha, library stats and builder_usd. Returns 0, or 1 when it failed or left problems."""
    sha = run.git(lay.repo, "rev-parse", f"{base}^{{commit}}").strip()
    for name in config.EDITABLE_FILES:
        shutil.rmtree(lay.repo / name, ignore_errors=True)
        run.git(lay.repo, "checkout", sha, "--", name)
    tree = lay.versions / f"{label}-base-{base}"
    if not tree.exists():
        materialize(ad.TaskStore().manifest(), sha, repo_root=lay.repo, dest=tree)
    agent = build_builder(lay, run_id=state["run_id"], pin=bpin, search_pin=pins.get("search"))
    outcome = None
    for attempt in (1, 2):
        if attempt == 2 and not cap_left():
            break
        print(f"{label}: builder on {base} (attempt {attempt}) ...", flush=True)
        outcome = agent.run(iter_n=iter_n, materialized_path=tree)
        rec["builder_usd"] = round(float(rec.get("builder_usd") or 0.0) + float(outcome.cost_usd or 0.0), 4)
        rec["builder_attempts"] = int(rec.get("builder_attempts") or 0) + 1
        rec["builder_detail"] = str(outcome.detail)[:300]
        save_state(lay, state)
        if outcome.exit_code == 0:
            break
        print(f"{label}: attempt {attempt} failed: {str(outcome.detail)[:200]}", flush=True)
    if outcome is None or outcome.exit_code != 0:
        print(f"STOPPED: {label}'s builder failed: {outcome.detail if outcome else 'not started'}")
        return 1
    rec["retired"] = retire_entries(lay.repo / "library", sub="examples")
    tag, new_sha = commit_new_version(ad.TaskStore().manifest(), f"{state['run_id']} {label}", repo_root=lay.repo,
                                      allow_empty=True)
    vtree = lay.versions / f"{label}-{tag}"
    materialize(ad.TaskStore().manifest(), new_sha, repo_root=lay.repo, dest=vtree)
    problems = library_version_problems(vtree, examples_allowed=(label != "discovery-director-plan"))
    if problems:
        rec.setdefault("refused_tags", []).append(tag)
        save_state(lay, state)
        print(f"STOPPED: {tag} has {len(problems)} problems, first: {problems[:3]}; re-running starts {label} again from {base}")
        return 1
    rec.update({"tag": tag, "version_sha": new_sha, "library": library_stats(vtree)})
    save_state(lay, state)
    print(f"{label}: {tag} {rec['library']}; builder ${rec['builder_usd']:.2f}", flush=True)
    return 0


def discovery_spent(state: dict) -> float:
    return round(sum(float(r.get("builder_usd") or 0) for r in state.get("discovery", {}).values()), 4)


def cmd_discover(args) -> int:
    """The library arm's discovery: DISCOVERY_ROLES in order, each a builder session on the previous one's version,
    stopping before a session that could pass config.DISCOVERY_BUDGET_USD. --stop-after N pauses after N sessions."""
    if not config.LIBRARY:
        raise SystemExit("discover needs LIBRARY=1")
    lay = layout()
    state = load_state(lay, args.run_id)
    disc = state.setdefault("discovery", {})
    pins, bpin = run.committed_pins(lay), builder_pin(lay)
    stage_brief(lay)
    base = "program-v0"
    for n, role in enumerate(DISCOVERY_ROLES, 1):
        rec = disc.setdefault(role, {"session": n, "role": role})
        if "tag" in rec:
            base = rec["tag"]
            continue
        if args.stop_after is not None and n > args.stop_after:
            print(f"PAUSED after {args.stop_after} discovery sessions (--stop-after); spent ${discovery_spent(state):.2f}")
            return 0

        def cap_left() -> bool:
            return discovery_spent(state) + BUILDER_BUDGET_USD <= config.DISCOVERY_BUDGET_USD

        if not cap_left():
            print(f"STOPPED: session {n} ({role}) could pass the ${config.DISCOVERY_BUDGET_USD:.0f} discovery cap "
                  f"(spent ${discovery_spent(state):.2f})")
            return 3
        ROLE.update(role=role, round=None)
        code = commit_builder_session(lay, state, rec, base=base, iter_n=n, label=f"discovery-{role}", bpin=bpin,
                                      pins=pins, cap_left=cap_left)
        if code:
            return code
        base = rec["tag"]
    state["discovery_final"] = {"tag": base, "version_sha": disc["director-synth"]["version_sha"],
                                "spent_usd": discovery_spent(state)}
    save_state(lay, state)
    print(json.dumps(state["discovery_final"]))
    return 0


def cmd_loop(args) -> int:
    lay = layout()
    state = load_state(lay, args.run_id)
    rounds = json.loads((config.DATA_DIR / "rounds.json").read_text())
    pins = run.committed_pins(lay)
    bpin = builder_pin(lay)
    per_scan = float(os.environ.get("READER_PER_SCAN_ESTIMATE_USD", "0.30"))
    stage_brief(lay)

    def total() -> float:
        return spent(state) + args.already_spent

    def reserve() -> tuple[float, float]:  # the guide arm's VAL pass still to come, at the measured cost per scan
        measured = [r for r in state["rounds"].values() if r.get("reader_sessions")]
        est = (sum(r["reader_usd"] for r in measured) / sum(r["reader_sessions"] for r in measured)) if measured else per_scan
        return est, 1.15 * est * args.val_scans

    first = 0 if config.ATLAS else 1  # the atlas arm's round 0 is the literature review: a builder, no reader
    if config.LIBRARY:  # spent includes discovery (the cap is the study's total through VAL)
        args.already_spent += discovery_spent(state)
    for k in range(first, config.ITERATIONS + 1):
        rec = state["rounds"].setdefault(str(k), {"round": k})
        iter_n = k + 1 if config.ATLAS else k
        # 1. the builder (library arm: round 1 reads discovery's library; from round 2 the director revises first)
        if config.LIBRARY and "tag" not in rec:
            if k == 1:
                fin = state.get("discovery_final")
                if not fin:
                    raise SystemExit("run discover first: no discovery library")
                rec.update({"tag": fin["tag"], "version_sha": fin["version_sha"]})
                save_state(lay, state)
            else:
                if total() + BUILDER_BUDGET_USD + reserve()[1] > args.cap:
                    print(f"STOPPED: round {k}'s director could pass the ${args.cap:.0f} cap (spent ${total():.2f})")
                    return 3
                ROLE.update(role="climb", round=k)
                code = commit_builder_session(lay, state, rec, base=state["rounds"][str(k - 1)]["tag"],
                                              iter_n=len(DISCOVERY_ROLES) + k - 1, label=f"climb-round{k}", bpin=bpin,
                                              pins=pins, cap_left=lambda: total() + BUILDER_BUDGET_USD <= args.cap)
                if code:
                    return code
        if "tag" not in rec:
            if total() + BUILDER_BUDGET_USD + reserve()[1] > args.cap:
                print(f"STOPPED: round {k}'s builder could pass the ${args.cap:.0f} cap with the guide arm's VAL pass "
                      f"(~${reserve()[1]:.2f}) kept back (spent ${total():.2f})")
                return 3
            base = state["rounds"][str(k - 1)]["tag"] if k > first else "program-v0"
            sha = run.git(lay.repo, "rev-parse", f"{base}^{{commit}}").strip()
            for name in config.EDITABLE_FILES:  # back to the base: a refused attempt may have left files behind
                shutil.rmtree(lay.repo / name, ignore_errors=True)
                run.git(lay.repo, "checkout", sha, "--", name)
            tree = lay.versions / f"loop-{args.run_id}-round{k}-base-{base}"
            if not tree.exists():
                materialize(ad.TaskStore().manifest(), sha, repo_root=lay.repo, dest=tree)
            agent = build_builder(lay, run_id=args.run_id, pin=bpin, search_pin=pins.get("search"))
            outcome = None
            for attempt in (1, 2):  # a failed session keeps nothing; one retry before the round stops
                if attempt == 2 and total() + BUILDER_BUDGET_USD + reserve()[1] > args.cap:
                    break
                print(f"round {k}: builder on {base} (attempt {attempt}) ...", flush=True)
                outcome = agent.run(iter_n=iter_n, materialized_path=tree)
                rec["builder_usd"] = round(float(rec.get("builder_usd") or 0.0) + float(outcome.cost_usd or 0.0), 4)
                rec["builder_attempts"] = int(rec.get("builder_attempts") or 0) + 1
                rec["builder_detail"] = str(outcome.detail)[:300]
                save_state(lay, state)
                if outcome.exit_code == 0:
                    break
                print(f"round {k}: builder attempt {attempt} failed: {str(outcome.detail)[:200]}", flush=True)
            if outcome is None or outcome.exit_code != 0:
                print(f"STOPPED: round {k}'s builder failed: {outcome.detail if outcome else 'not started'}")
                return 1
            rec["retired"] = retire_entries(lay.repo / ("atlas" if config.ATLAS else "examples"))
            tag, new_sha = commit_new_version(ad.TaskStore().manifest(), f"loop {args.run_id} round {k}",
                                              repo_root=lay.repo, allow_empty=True)
            vtree = lay.versions / f"loop-{args.run_id}-round{k}-{tag}"
            materialize(ad.TaskStore().manifest(), new_sha, repo_root=lay.repo, dest=vtree)
            allowed = {e["scan"]: e["cause"] for e in json.loads((scans_dir(lay) / "index.json").read_text())}
            problems = (atlas_version_problems(vtree, allowed, literature_only=k == 0) if config.ATLAS
                        else version_problems(vtree, allowed))
            if problems:
                rec.setdefault("refused_tags", []).append(tag)
                save_state(lay, state)
                print(f"STOPPED: {tag} has {len(problems)} problems, first: {problems[:3]}; re-running starts round "
                      f"{k} again from {base}")
                return 1
            rec.update({"tag": tag, "version_sha": new_sha, "library": library_stats(vtree)})
            save_state(lay, state)
            print(f"round {k}: {tag} {rec['library']}; builder ${rec['builder_usd']:.2f}", flush=True)
        # 2. the reader
        if k == 0:
            continue
        if "mean_auroc" not in rec:
            ids = sorted(rounds[str(k)])
            est, keep = reserve()
            if total() + 1.15 * est * len(ids) + keep > args.cap:
                print(f"STOPPED: round {k}'s reader pass (~${est * len(ids):.2f}) could pass the ${args.cap:.0f} cap "
                      f"with the guide arm's VAL pass (~${keep:.2f}) kept back (spent ${total():.2f})")
                return 3
            tag = rec["tag"]
            tree = lay.versions / f"loop-{args.run_id}-round{k}-{tag}"
            if not tree.exists():
                materialize(ad.TaskStore().manifest(), rec["version_sha"], repo_root=lay.repo, dest=tree)
            agentic.set_tree(tree)
            inputs = ad.TaskSource(lay.private / "loop-batches").stage(ids, "train", f"loop-{args.run_id}-round{k}-n{len(ids)}")
            pass_run = f"loop-{args.run_id}-round{k}"
            print(f"round {k}: reader on {len(ids)} TRAIN scans with {tag} ...", flush=True)
            runner = run.program_runner(lay, image=run.images()[1], pin=pins["program"])
            with runner:
                artifacts = runner.as_runner(run_id=pass_run).run(tree, inputs)
            rec["reader_usd"], rec["reader_sessions"] = pass_cost(pass_run)
            save_state(lay, state)
            if artifacts.status != "ok":
                print(f"STOPPED: round {k}'s reader pass failed: {artifacts.status} {artifacts.error}")
                return 1
            score = ad.TaskScorer().score(artifacts, "train", {})
            summary, mistakes = write_feedback(lay, k, artifacts, score, tree)
            added = 0 if config.LIBRARY else add_scans(lay, mistakes, f"round-{k}")  # the library's director sees all TRAIN
            rec.update({"answers": artifacts.artifact_refs[0].path, "mean_auroc": summary["mean_auroc"],
                        "auroc": summary["auroc"], "ranked_first_right": summary["ranked_first_right"],
                        "share_opened_all_drawn_images": summary["share_opened_all_drawn_images"],
                        "share_read_guide": summary["share_read_guide"], "failed_sessions": summary["failed_sessions"],
                        **{x: summary.get(x) for x in ("share_used_library", "mean_library_commands")},
                        "mistakes_added": added, "finished": now()})
            save_state(lay, state)
            print(f"round {k}: TRAIN mean {summary['mean_auroc']:.3f} {summary['auroc']}; opened all drawn images "
                  f"{summary['share_opened_all_drawn_images']:.0%}; reader ${rec['reader_usd']:.2f} for "
                  f"{rec['reader_sessions']}; spent ${total():.2f}", flush=True)
            if k == 1 and not config.LIBRARY and summary["share_opened_all_drawn_images"] < 0.5:
                print("STOPPED: fewer than half the round-1 reader sessions opened every drawn image")
                return 3
        if args.stop_after is not None and k >= args.stop_after and k < config.ITERATIONS:
            print(f"PAUSED after round {k} (--stop-after); re-run without it to continue")
            return 0
    state["final"] = {"round": config.ITERATIONS, "tag": state["rounds"][str(config.ITERATIONS)]["tag"]}
    state["spent_usd"] = spent(state)
    save_state(lay, state)
    print(json.dumps({"final": state["final"], "spent_usd": state["spent_usd"]}))
    return 0


def cmd_retool(args) -> int:
    """Commit the seed program's reader instructions and tools over the run's latest library version (a host commit,
    as the KB pilot's retag did), and point the run at it: the last finished discovery session's (or climb round's)
    tag becomes the new version, so the next session builds on it. For a fixed-tools change mid-run (2026-11-25: the
    library text limit). Prints the new tag."""
    lay = layout()
    state = load_state(lay, args.run_id)
    if args.from_tag:  # a follow-up arm on a named version; the run's records stay as they are with --no-state
        rec = {"tag": args.from_tag}
    else:
        recs = [r for r in state.get("discovery", {}).values() if r.get("tag")]
        recs += [r for _k, r in sorted(state.get("rounds", {}).items(), key=lambda kv: int(kv[0])) if r.get("tag")]
        if not recs:
            raise SystemExit("no finished version to retool")
        rec = recs[-1]
    sha = run.git(lay.repo, "rev-parse", f"{rec['tag']}^{{commit}}").strip()
    shutil.rmtree(lay.repo / "library", ignore_errors=True)
    run.git(lay.repo, "checkout", sha, "--", "library")
    for name in config.CONTRACT_FILES:
        dest = lay.repo / name
        if dest.is_dir():
            shutil.rmtree(dest)
        src = config.SEED_PROGRAM_DIR / name
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dest)
    shutil.rmtree(lay.repo / "tools" / "__pycache__", ignore_errors=True)
    tag, new_sha = commit_new_version(ad.TaskStore().manifest(), f"{args.run_id} retool on {rec['tag']}: {args.why}",
                                      repo_root=lay.repo, allow_empty=True)
    if args.no_state:
        state.setdefault("follow_ups", []).append({"from": rec["tag"], "tag": tag, "version_sha": new_sha, "why": args.why})
        save_state(lay, state)
        print(f"retooled {rec['tag']} -> {tag} (run records unchanged)")
        return 0
    rec.setdefault("retooled_from", []).append(rec["tag"])
    rec.update({"tag": tag, "version_sha": new_sha})
    if state.get("discovery_final") and state["discovery_final"].get("tag") in rec["retooled_from"]:
        state["discovery_final"].update({"tag": tag, "version_sha": new_sha})
    save_state(lay, state)
    print(f"retooled {rec['retooled_from'][-1]} -> {tag}")
    return 0


def cmd_report(args) -> int:
    state = load_state(layout(), args.run_id)
    rows = []
    for k in sorted(state["rounds"], key=int):
        r = state["rounds"][k]
        rows.append({x: r.get(x) for x in ("round", "tag", "library", "mean_auroc", "auroc", "ranked_first_right",
                                           "share_opened_all_drawn_images", "share_read_guide", "failed_sessions",
                                           "share_used_library", "mean_library_commands",
                                           "mistakes_added", "builder_usd", "reader_usd", "reader_sessions")})
    out = {"run_id": args.run_id, "rounds": rows, "final": state.get("final"), "spent_usd": spent(state)}
    if state.get("discovery"):
        out["discovery"] = [{x: r.get(x) for x in ("session", "role", "tag", "library", "builder_usd", "retired")}
                            for r in state["discovery"].values()]
        out["discovery_final"] = state.get("discovery_final")
        out["spent_usd"] = round(out["spent_usd"] + discovery_spent(state), 4)
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_print_prompt(args) -> int:
    from isolation.contained_agent import INPUTS_PATH, OUT_PATH, OptimizerPaths

    names_in = (op.TASK_BRIEF_NAME, op.NOTEBOOK_NAME, "scans", "reader")
    paths = OptimizerPaths(inputs={n: f"{INPUTS_PATH}/{n}" for n in names_in},
                           outputs={n: f"{OUT_PATH}/{n}" for n in (op.CHANGE_NOTE_NAME, op.NOTEBOOK_NAME)},
                           notes="/workspace/notes")
    print(op.system_prompt())
    print("=" * 80)
    print(builder_prompt(args.iter, paths))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("pins")
    p.add_argument("--write", action="store_true")
    lp = sub.add_parser("loop")
    lp.add_argument("--run-id", default="loop-r1", type=run.run_id_arg)
    lp.add_argument("--cap", type=float, default=config.RUN_BUDGET_USD, help="the study's total through VAL (spec)")
    lp.add_argument("--val-scans", type=int, default=134, help="the guide arm's VAL pass to keep room for")
    lp.add_argument("--already-spent", type=float, default=0.0, help="dollars spent before the loop (launcher)")
    lp.add_argument("--stop-after", type=int, default=None, help="pause after this round's reader pass")
    dp = sub.add_parser("discover")
    dp.add_argument("--run-id", default="library-r1", type=run.run_id_arg)
    dp.add_argument("--stop-after", type=int, default=None, help="pause after this many discovery sessions")
    rp = sub.add_parser("report")
    rp.add_argument("--run-id", default="loop-r1")
    rp.add_argument("--out")
    rt = sub.add_parser("retool")
    rt.add_argument("--run-id", default="library-r1", type=run.run_id_arg)
    rt.add_argument("--why", required=True)
    rt.add_argument("--from-tag", help="retool this version instead of the run's latest")
    rt.add_argument("--no-state", action="store_true", help="leave the run's records unchanged (a follow-up arm)")
    pp = sub.add_parser("print-prompt")
    pp.add_argument("--iter", type=int, default=2)
    args = ap.parse_args(argv)
    return {"init": cmd_init, "pins": cmd_pins, "loop": cmd_loop, "report": cmd_report, "discover": cmd_discover,
            "print-prompt": cmd_print_prompt, "retool": cmd_retool}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
