"""The agentic program mode: the program is an instructions file (plus tools), and each patient is read by its
own sealed Claude Code session that follows it. The engine's ``ProgramRunner`` runs the sessions; this file
says what one call is.

Each call: the patient's record folder (the shared exporter's layout: one parquet per OMOP table, ``notes.txt``,
``registry_<table>.parquet``, ``meta.json``; everything cut at first treatment - 7 days) is copied, files read-only,
to ``record/`` in the call's own folder (its only writable mount); the session reads ``instructions.md`` from the
read-only program, writes a short findings note, then ``output.json`` with two forced answers, yes or no: did the
patient die within 1 year, and within 2 years, of diagnosis; and a reason. With ``config.PROBABILITY`` the two
answers are instead a 0-100 chance at each horizon (stored as 0..1), plus up to ``MAX_REASONS`` reasons: the record
facts that moved the probability most, each naming the record file it came from and its date (Phil 2026-10-31).
With ``config.SURVIVAL`` the answer is a survival curve from the record's last day, as points (months, chance alive),
plus the same reasons (Phil 2026-11-01); ``curve_of`` decides what is readable.
``summarize`` writes the same ``{"answers": {...}}`` file the rules-based runner writes (one yes/no per horizon,
True/False, or None when unreadable), so the scorer, the mistake builder and ``run.py test`` read one format.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import config
from adapters import patient_dir, rows_for

ENTRY = "instructions.md"
OUTPUT = "output.json"
#: Where each kind of fact lives in a patient's folder, adapted from the Cohort Explorer's guides (Phil 2026-10-28:
#: the session must be told what it can read and where to look). Fixed, like the wrapper: staged read-only beside
#: the record for every call, so no optimized version can drop it. The optimizer's brief points at the same file.
GUIDE = "record_guide.md"
GUIDE_SRC = Path(__file__).resolve().parent / "context" / "patient_data_guide.md"
#: Part of the key that decides whether a finished pass can be reused. Bump it whenever you change
#: ``prompt`` below, or a pass run under the old wording is served again. The guide's hash is part of it, so an
#: edit to the guide also stops old passes being served.
PROMPT_TAG = (f"died-2yr-ehr-v3-guide-{hashlib.sha256(GUIDE_SRC.read_bytes()).hexdigest()[:10]}"
              + ("-summaryonly-bare-v1" if config.SUMMARY_ONLY else "-summary-v1" if config.WITH_SUMMARY else "")
              + ("-surv-v1" if config.SURVIVAL else "-prob-v1" if config.PROBABILITY else ""))
#: Probability mode: at most this many reasons are kept per answer (the prompt asks for 3 to 5).
MAX_REASONS = 5
#: The patient summary's name in the session's record folder (config.WITH_SUMMARY only).
SUMMARY = "summary.txt"
#: The session's reasoning is not kept in its transcript (thinking blocks arrive empty), so it writes what it
#: saw; the optimizer reads it beside the transcript for TRAIN mistakes.
FINDINGS = "findings.md"
#: No Agent tool: a subagent would read the record again at full price. No network but Claude; nothing mounted
#: but the program and this call's folder.
TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")
#: The session's copy of the record, inside its call folder.
RECORD = "record"
#: The share of TRAIN patients who died within each horizon (718 and 1,188 of 3,886), stated in the prompt (Phil
#: 2026-10-21: the cohort's death rate). From splits/train.jsonl; scripts/build_data.py prints it.
TRAIN_DEATH_RATE = {"1 year": "18.5%", "2 years": "30.6%"}
#: Survival mode: TRAIN's Kaplan-Meier survival from the record's last day (3,836 TRAIN patients with an os_survival
#: time), stated in the prompt as the cohort's average curve. scripts/check_task.py recomputes it from the key.
TRAIN_SURVIVAL = {"1 year": "80.2%", "2 years": "68.5%", "3 years": "61.2%", "5 years": "51.2%", "10 years": "32.8%"}
#: Survival mode: the curve the prompt asks for runs to this month; a curve is unreadable only if it stops before
#: adapters.cover_months() (TRAIN's scoring window end, about 74.6 months).
ASK_MONTHS = 120
#: Survival mode: at most this many points in a curve.
MAX_POINTS = 2000
#: The OMOP tables the exporter writes (no death table), in the order the prompt lists them.
OMOP_TABLES = ("person", "observation_period", "visit_occurrence", "visit_detail", "condition_occurrence",
               "procedure_occurrence", "drug_exposure", "drug_era", "device_exposure", "episode", "image_occurrence",
               "measurement", "observation", "note")
REGISTRY_TABLES = ("diagnoses", "treatments", "outcomes", "events", "misc_details")
#: Every file a session may be given. The exporter also writes OUT/registry_dates/ (the raw governing dates its
#: leak check needs, i.e. the very dates the allow-list hides); that folder sits beside patients/, and this list is
#: what keeps it, or anything else stray, out of a session even if it ever lands in a patient folder.
RECORD_FILES = frozenset({f"{t}.parquet" for t in OMOP_TABLES} | {f"registry_{t}.parquet" for t in REGISTRY_TABLES}
                         | {"notes.txt", "meta.json"})


def record_files(src: Path) -> list[Path]:
    """The files of one patient folder that may be staged, after checking the folder holds nothing else.
    Raises on a folder outside DATA_DIR/patients/, a subfolder, or any file not in RECORD_FILES."""
    src = Path(src)
    if src.resolve().parent != (config.DATA_DIR / "patients").resolve():
        raise RuntimeError("a record folder must be DATA_DIR/patients/<id>")
    files = sorted(src.iterdir())
    stray = [f for f in files if f.is_dir() or f.name not in RECORD_FILES]
    if stray:
        raise RuntimeError(f"refusing to stage a record folder with {len(stray)} unexpected entries "
                           f"(first: {stray[0].name}); only the exporter's patient files may reach a session")
    return files


def yes_no(value) -> bool | None:
    """True for "yes", False for "no" (any case, surrounding spaces ignored), None for anything else: a forced
    answer is one of the two words, and anything else is scored wrong (eval-set spec, Decided: Score)."""
    word = value.strip().lower() if isinstance(value, str) else None
    return {"yes": True, "no": False}.get(word)


def percent(value) -> float | None:
    """Probability mode: a stated chance 0-100 (a number, not a word or a bool) as 0..1, or None when unreadable or
    out of range. A failed answer is scored at the middle of the ranking (adapters.FAILED_PROBABILITY)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value / 100 if 0 <= value <= 100 else None


def curve_of(value, cover: float) -> tuple[list[list[float]] | None, str | None]:
    """Survival mode: a stated curve as [[month, survival], ...], or (None, why) when unreadable. Readable means 1 to
    MAX_POINTS points, each two finite numbers (not bools), months from 0 and strictly increasing, survival within
    0-1 and never rising, and the last point at or past ``cover`` months. A point may also be {"month", "survival"}."""
    if not isinstance(value, list) or not 0 < len(value) <= MAX_POINTS:
        return None, "not_a_curve"
    pts = []
    for p in value:
        if isinstance(p, dict):
            p = [p.get("month"), p.get("survival")]
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in p)):
            return None, "not_a_curve"
        pts.append([float(p[0]), float(p[1])])
    if pts[0][0] < 0 or any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
        return None, "not_a_curve"
    if any(not 0 <= y <= 1 for _m, y in pts):
        return None, "outside_0_1"
    if any(b[1] > a[1] + 1e-9 for a, b in zip(pts, pts[1:])):
        return None, "rising_curve"
    if pts[-1][0] < cover:
        return None, "too_short"
    return pts, None


def repair_curve(value):
    """Read the obvious intent of a step survival curve, fixing the cosmetic slips a session sometimes makes so one
    bad answer in a pass does not throw the whole pass away (Phil 2026-11-02): drop a point whose month is not above
    the last kept one (a duplicate or out-of-order month), clamp each survival into 0-1, and make the curve never
    rise (each survival is the lower of itself and the one before). It does NOT guess at substance: if any point is
    not two finite numbers, or the value is not a non-empty list, it is returned unchanged so curve_of still rejects
    a genuinely broken or absent curve (and the run still stops). Monotone, in-range curves pass through unchanged."""
    if not isinstance(value, list) or not value:
        return value
    pts = []
    for p in value:
        if isinstance(p, dict):
            p = [p.get("month"), p.get("survival")]
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in p)):
            return value  # a malformed point: don't guess, let curve_of reject the whole thing
        pts.append([float(p[0]), float(p[1])])
    cleaned, last_m, run = [], -math.inf, 1.0
    for m, s in pts:
        if m <= last_m:
            continue
        run = min(run, min(1.0, max(0.0, s)))
        cleaned.append([m, run])
        last_m = m
    return cleaned


def loads_lenient(raw: str) -> dict | None:
    """Read a session's output.json, and if it is not quite valid JSON, recover what is scored anyway: a typo in the
    reasons (which are never scored) must not lose the survival curve. On a parse failure, pull the survival_curve
    array out by matching its brackets and read pass_id with a regex. Returns a dict (whole object when valid, or just
    the recovered keys), or None when nothing usable is there (the caller then scores it unreadable)."""
    try:
        return json.loads(raw)
    except ValueError:
        pass
    out: dict = {}
    key = raw.find('"survival_curve"')
    if key != -1:
        start = raw.find("[", key)
        if start != -1:
            depth = 0
            for j in range(start, len(raw)):
                if raw[j] == "[":
                    depth += 1
                elif raw[j] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            out["survival_curve"] = json.loads(raw[start:j + 1])
                        except ValueError:
                            pass
                        break
    m = re.search(r'"pass_id"\s*:\s*"([^"]*)"', raw)
    if m:
        out["pass_id"] = m.group(1)
    return out or None


def reasons_of(value) -> list[dict]:
    """Probability mode: the stated reasons, kept as {"fact", "file", "date"} strings (anything else in an item is
    dropped, an item that is not an object is skipped); never scored. scripts/check_reasons.py checks them."""
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            out.append({k: str(item.get(k) or "")[:300] for k in ("fact", "file", "date")})
    return out[:MAX_REASONS]


#: config.SUMMARY_ONLY: the eval set's summary-arm opening for Astra/Argon/MedGemma, verbatim (vista_eval_vlm main
#: src/evalset/cards/death.py summary_opening); scripts/check_task.py checks it against that file when it is there.
BARE_OPENING = ("You are given a summary of this patient's record, written from the record up to one week before "
                "first treatment.")


def bare_prompt(call, paths, pass_id: str) -> str:
    """config.SUMMARY_ONLY: the eval set's summary-arm prompt (opening, PATIENT SUMMARY, Question), with only the
    answer changed from yes/no to a probability (or, for survival, a curve), plus the minimum a session needs to run
    at all: where the program and its tools are, that the summary is also a file, and where to write. No cohort
    numbers, no folder layout, no record guide, no hints (Phil 2026-11-01)."""
    summary = (config.SUMMARY_DIR / f"{call.payload['id']}.txt").read_text(encoding="utf-8").strip()
    if config.SURVIVAL:
        question = ("Question: How likely is this patient to be alive, from any cause of death, at each point in the "
                    "years after the last day of their record? Give a survival curve.")
        answer_rules = (
            f'{{"survival_curve": [[<months after the record\'s last day>, <chance alive then, 0 to 1>], ...], '
            f'"reasons": [{{"fact": "<one fact from the summary>", "date": "<its date, YYYY-MM-DD, if it has one>"}}, '
            f'...], "pass_id": "{pass_id}"}}. The curve is points: months from 0 and strictly increasing, the chance '
            f"never rising, every chance between 0 and 1, running to at least month {ASK_MONTHS}, up to {MAX_POINTS} "
            f"points. It is read as a step curve. A curve that rises, leaves 0 to 1, or stops early counts as no "
            f"answer. In \"reasons\" list the 3 to 5 facts that shaped the curve most.")
    else:
        keys = ", ".join(f'"{key}": <a whole number from 0 to 100>' for key, _gold, _words in config.HORIZONS)
        question = ("Question: Will this patient die, from any cause, within 1 year of their cancer diagnosis? And "
                    "within 2 years?")
        answer_rules = (
            f'{{{keys}, "reasons": [{{"fact": "<one fact from the summary>", "date": "<its date, YYYY-MM-DD, if it '
            f'has one>"}}, ...], "pass_id": "{pass_id}"}}. Each number is your probability in percent, 0 to 100; a '
            f"number outside that range, or a word, counts as no answer. In \"reasons\" list the 3 to 5 facts from the "
            f"summary that moved your probabilities most.")
    return (
        f"Follow the instructions in {paths.program}/{ENTRY}; any tools are in {paths.program}/tools/ (run Python "
        f"with python3 -B). The summary below is also the file {paths.call}/{SUMMARY}.\n\n"
        f"{BARE_OPENING}\n\nPATIENT SUMMARY:\n{summary}\n\n"
        f"{question}\n\n"
        f"Before answering, write {paths.call}/{FINDINGS}: a short note of what you used, with no names, dates or "
        f"other identifiers. Then write {paths.call}/{OUTPUT} containing exactly one JSON object on one line: "
        + answer_rules
    )


def program_spec(forbidden):
    from isolation.pass_outputs import Call
    from isolation.program_runner import ProgramSpec

    def calls_for(inputs):
        return [Call(row["id"], {"id": row["id"]}) for row in rows_for(inputs)]  # never the label

    def stage(call, folder, pass_id):
        # Files read-only so the session cannot edit its evidence; the folder stays writable so a resume can
        # clear it.
        if config.SUMMARY_ONLY:  # the summary and nothing else: no record, no OMOP tables, no guide
            # Leave the call folder's permissions as the runner set them: on Vertex it chmods the folder 0o777 so the
            # container user (the credential's owner, not the host user) can write its output here. Chmod'ing it to
            # 0o755 stripped that write and the session refused to start (live cost check 2026-11-01). The record
            # branch below likewise never touches the folder itself, only the files/subfolders it stages read-only.
            shutil.copy2(config.SUMMARY_DIR / f"{call.payload['id']}.txt", Path(folder) / SUMMARY)
            (Path(folder) / SUMMARY).chmod(0o444)
            return
        dest = Path(folder) / RECORD
        dest.mkdir(parents=True)
        # Readable by others: on Vertex the container runs as the credential's owner, not the host user, and a
        # 077 umask left the record unreadable there (live check 2026-10-21).
        dest.chmod(0o755)
        for f in record_files(patient_dir(call.payload["id"])):
            shutil.copy2(f, dest / f.name)
            (dest / f.name).chmod(0o444)
        if config.WITH_SUMMARY:
            # Raises when a patient has no summary: the arm is only meaningful if every patient gets one.
            shutil.copy2(config.SUMMARY_DIR / f"{call.payload['id']}.txt", dest / SUMMARY)
            (dest / SUMMARY).chmod(0o444)
        shutil.copy2(GUIDE_SRC, Path(folder) / GUIDE)
        (Path(folder) / GUIDE).chmod(0o444)

    def prompt(call, stage_name, paths, pass_id):
        # The per-call wrapper: not optimized. Container paths only. The facts every version needs (the question,
        # the cutoff, the folder layout, the death rate) live here; the method lives in instructions.md.
        rec = f"{paths.call}/{RECORD}"
        if config.SUMMARY_ONLY:  # the bare eval-set summary prompt, answer changed to a probability/curve
            return bare_prompt(call, paths, pass_id)
        if config.SURVIVAL:
            question = (
                f"The question: how likely is this patient to be alive, from any cause of death, at each point in the "
                f"years after the last day of their record? Give it as a survival curve.")
            cohort = ("Of the patients in this group, counted from that same day, "
                      + ", ".join(f"{v} were alive at {k}" for k, v in TRAIN_SURVIVAL.items()) + ".")
            answer_rules = (
                f'one line: {{"survival_curve": [[<months after the record\'s last day>, <chance alive then, 0 to 1>], '
                f'...], "reasons": [{{"fact": "<one fact from the record>", "file": "<the record file it is in, e.g. '
                f'notes.txt or measurement.parquet>", "date": "<its date, YYYY-MM-DD>"}}, ...], "pass_id": "{pass_id}"}}. '
                f"The curve is points: months from 0 and strictly increasing, the chance never rising from one point "
                f"to the next, every chance between 0 and 1, running to at least month {ASK_MONTHS}, as many points "
                f"as you like up to {MAX_POINTS} (you may compute them with code). It is read as a step curve: between "
                f"two points the chance is the earlier point's, and 1 before the first. A curve that rises, leaves 0 "
                f"to 1, or stops early counts as no answer. In \"reasons\" list the 3 to 5 facts from the record that "
                f"shaped the curve most, most important first, each with the file you read it in and the date it "
                f"carries (every one is on or before the record's last day).")
        elif config.PROBABILITY:
            keys = ", ".join(f'"{key}": <a whole number from 0 to 100>' for key, _gold, _words in config.HORIZONS)
            ask = ("Give each as the chance, from 0 to 100, that it happened.")
            answer_rules = (
                f'one line: {{{keys}, "reasons": [{{"fact": "<one fact from the record>", "file": "<the record file '
                f'it is in, e.g. notes.txt or measurement.parquet>", "date": "<its date, YYYY-MM-DD>"}}, ...], '
                f'"pass_id": "{pass_id}"}}. Each number is your probability in percent, 0 to 100; a number outside '
                f"that range, or a word, counts as no answer. In \"reasons\" list the 3 to 5 facts from the record "
                f"that moved your probabilities most, most important first, each with the file you read it in and "
                f"the date it carries (every one is on or before the record's last day).")
        else:
            keys = ", ".join(f'"{key}": "yes" or "no"' for key, _gold, _words in config.HORIZONS)
            ask = "Answer each yes or no."
            answer_rules = (
                f'one line: {{{keys}, "reason": "<one line: the findings you relied on>", "pass_id": "{pass_id}"}}. '
                f"Each answer must be exactly yes or no: no probabilities, no hedging; an answer that is not one of the "
                f"two words is scored as wrong.")
        if not config.SURVIVAL:
            question = (f"The questions: did this patient die, of any cause, within 1 year of their cancer diagnosis? "
                        f"And within 2 years? {ask}")
            cohort = (f"Of the patients in this group, {TRAIN_DEATH_RATE['1 year']} died within 1 year and "
                      f"{TRAIN_DEATH_RATE['2 years']} within 2 years.")
        return (
            f"Follow the instructions in {paths.program}/{ENTRY}; any tools are in {paths.program}/tools/ (run "
            f"Python with python3 -B).\n\n"
            f"{question} The patient has a thoracic cancer. Their medical record is the folder "
            f"{rec}/ and it stops 7 days before their first cancer treatment: nothing dated after that day is in it. "
            f"{cohort}\n\n"
            f"The folder (read-only):\n"
            f"- notes.txt: every clinical note in date order, each starting with a header line that begins "
            f"'===== NOTE'. Search it with grep.\n"
            f"- one parquet file per table of the OMOP common data model: {', '.join(OMOP_TABLES)}. A table with "
            f"no rows for this patient may be absent.\n"
            f"- {', '.join(f'registry_{t}.parquet' for t in REGISTRY_TABLES)}: the cancer registry's record of "
            f"this cancer, limited to the fields allowed at that cutoff.\n"
            f"- meta.json: the patient's opaque id and the cutoff date.\n"
            + (f"- {SUMMARY}: a summary of this record written earlier by a language model from the same record, "
               f"cut at the same day. It can be wrong or leave things out; check what matters in the record.\n"
               if config.WITH_SUMMARY else "") +
            f"python3 has pandas, pyarrow and duckdb for the parquet files.\n\n"
            f"Read {paths.call}/{GUIDE} before the record: it says what each file holds, where each kind of fact "
            f"(stage, histology, biomarkers, treatment, imaging and reports, labs, smoking, visits) lives, how the "
            f"codes read, and which sources mislead.\n\n"
            f"Scratch files go in /tmp. Before answering, write {paths.call}/{FINDINGS}: "
            f"a short note of what you looked at and what you found that bears on the answer. Use no names, dates "
            f"or other identifiers in it. Then write {paths.call}/{OUTPUT} containing exactly one JSON object on "
            + answer_rules
        )

    def read_stamp(call, folder):
        try:
            raw = (Path(folder) / OUTPUT).read_text(encoding="utf-8")
        except OSError:
            return None
        body = loads_lenient(raw)
        return body.get("pass_id") if isinstance(body, dict) else None

    def summarize(archive, receipts):
        answers = {}
        for r in receipts:
            said, reason, error = ({"curve": None} if config.SURVIVAL else {key: None for key, _g, _w in config.HORIZONS},
                                   None, r.get("failed"))
            reasons: list[dict] = []
            out = next((o["path"] for o in r.get("outputs", []) if o["rel"] == OUTPUT), None)
            log = next((o["path"] for o in r.get("outputs", []) if o["rel"] == FINDINGS), None)
            if out and not error:
                try:
                    _raw = Path(out).read_text(encoding="utf-8")
                    body = loads_lenient(_raw) if config.SURVIVAL else json.loads(_raw)
                    if body is None:
                        raise ValueError("no recoverable output")
                    if config.SURVIVAL:
                        from adapters import cover_months

                        curve, why = curve_of(repair_curve(body.get("survival_curve")), cover_months())
                        said = {"curve": curve}
                        reasons = reasons_of(body.get("reasons"))
                        reason = "; ".join(r["fact"] for r in reasons)[:500]
                        error = why
                    elif config.PROBABILITY:
                        said = {key: percent(body.get(key)) for key, _g, _w in config.HORIZONS}
                        reasons = reasons_of(body.get("reasons"))
                        reason = "; ".join(r["fact"] for r in reasons)[:500]
                        if None in said.values():
                            error = "not_a_probability"
                    else:
                        said = {key: yes_no(body.get(key)) for key, _g, _w in config.HORIZONS}
                        reason = str(body.get("reason") or "")[:500]
                        if None in said.values():
                            error = "not_yes_or_no"
                except (OSError, ValueError, AttributeError):
                    error = "unreadable_output"
            elif not error:
                error = "no_output"
            answers[r["call_id"]] = {**said, "reason": reason, "error": error, "findings_log": log,
                                     **({"reasons": reasons} if config.PROBABILITY else {})}
        manifest = Path(archive) / "answers.json"
        manifest.write_text(json.dumps({"answers": answers}), encoding="utf-8")
        return manifest

    return ProgramSpec(
        calls_for=calls_for, stage=stage, prompt=prompt, read_stamp=read_stamp, summarize=summarize,
        stages=("label",), output_patterns=(OUTPUT, FINDINGS), prompt_tag=PROMPT_TAG,
        max_concurrent=config.PROGRAM_MAX_CONCURRENT, timeout_seconds=config.PROGRAM_CALL_TIMEOUT_SECONDS,
        max_budget_usd=config.PROGRAM_CALL_BUDGET_USD, forbidden=forbidden,
        # On Vertex the sessions use the mounted optimizer login; the seat token is read only on the seat route.
        token_env_name=None if config.ROUTE.is_vertex else config.TOKEN_ENV_NAME,
    )


def program_runner(layout, *, image: str, pin: str, forbidden):
    """The engine's ProgramRunner over this spec. Pass it to the driver as is; the driver enters it."""
    from isolation import sealed_session as ss
    from isolation.program_runner import ProgramRunner
    from isolation.session_scope import SessionScope

    profile, credential = config.ROUTE.session_auth()
    template = ss.SessionDescription(
        # A placeholder grant: the runner builds each call's real one. denied may not be empty.
        scope=SessionScope(program=layout.versions, denied=forbidden),
        posture="deny-by-default", tools=TOOLS, allowed_tools=TOOLS,
        model=config.PROGRAM_MODEL, image=image,
        profile=profile, credential=credential,
        transcript_dir=layout.transcripts / "program", expected_fingerprint=pin,
    )
    return ProgramRunner(spec=program_spec(forbidden), template=template, output_root=layout.program_outputs,
                         materialize_root=layout.versions)
