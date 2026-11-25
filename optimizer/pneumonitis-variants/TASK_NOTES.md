# pneumonitis-variants: task notes

Spec: `planning/vista_eval_vlm/2026-11-24-ct-pneumonitis-variants.md` on the mount (Approved 2026-11-24). Built
2026-11-24/25 on branch `feat/ct-pneumonitis-variants` off `feat/ct-organ-zoom` @6c1d699, from ct-findings-5's organ-zoom
tool, knowledge-base loop and pre-drawn examples. Engine pin agentic-label-opt bb216c2 (as ct-findings-5).

## The idea

A Sonnet reader, one sealed session per chest CT, gives a 0-100 probability for each of five causes of the lung
change: radiation, immunotherapy, other drug, organizing pneumonia (COP), infection. An Opus builder, with the engine's
search proxy (Radiopaedia and the literature as text and links), writes the reader's `guide/` and `examples/` over a
5-round loop on TRAIN; the reader's instructions and tools never change. program-v0 (empty guide and library) is the
zero-shot arm, run on VAL first; round 5's version is the guide arm. Win = the paired guide - zero-shot interval on
the five-variant mean AUROC above zero.

## Decisions taken at build time

| Decision | Answer |
|---|---|
| Infection labels | Gemini 2.5 Pro on Vertex (BAA project), one call per report, yes/no/unsure; yes only (spec). The wording match without its negation filter gives exactly the grill's 545 reports; negation is left to the model (with a regex filter: 542-543) |
| Infection set | confirmed yes or the existing model label `infectious` (strong/weak), minus benchmark pneumonitis positives (3 TRAIN): those keep their variant |
| Scans | all eligible scans, several per patient allowed; the patient bootstrap and the example pick group by patient |
| TRAIN draw | 60 source scans for the builder (12 per variant, one per patient), then 5 rounds of 80 (16 per variant) from other patients; COP and immunotherapy run out and are reused from earlier rounds (61 and 45 reused) |
| Example pick | per variant, the 2 entries whose source scan is most similar (MerlinOnc cosine), never the case's own patient; drawn beside the same lung region (first view of the entry) |
| Builder mistakes | a reader scan whose highest-probability cause is not its answer (or failed): added to the builder's scans with CT and report |
| Early stop | after round 1, under half the reader sessions opening every drawn image stops the loop (an earlier form of the spec's guide cost-check rule, which also runs) |
| Image | ct-findings-5's optimizer image (same Dockerfile) |
| Host staging | the engine's environment has no pillow or numpy: every launcher command adds `--with pillow --with numpy` (as ct-findings-5's prestage launchers) |

## Data (built 2026-11-24)

- Infection pass: 545 reports, yes 407 / no 42 / unsure 96, $3.74. Infection scans: TRAIN 192, VAL 40, TEST 33 (TEST not used).
- VAL 134 scans, 84 patients: radiation 28, immunotherapy 16, other drug 32, COP 18, infection 40. Every eligible scan had a volume and a mask.
- TRAIN 599 eligible (probe); 349 distinct in the source and rounds (1 round-2 scan dropped 2026-11-25: its organ mask has no lungs; round 2 has 79).
- MerlinOnc probe on the 134 VAL scans (scripts/score_probe.py): radiation 0.875, immunotherapy 0.768, other drug 0.772, COP 0.737, infection 0.773; mean 0.785. 30-shuffle null p95 0.61-0.72.

## Commands

    uv run -q --no-project --with google-cloud-bigquery --with google-genai python -B scripts/infection_pass.py   ~$4
    scripts/build_data.py select | volumes | masks | reports | check          (see its docstring)
    uv run -q --no-project --with numpy --with scikit-learn python -B scripts/score_probe.py                      $0
    uv run -q --project $AGENTIC_LABEL_OPT --with pytest --with numpy --with pillow --with pandas python -m pytest -q tests/
    bash scripts/run_variants.sh                                                   PAID, <= $250 (the spec's approved run)

## Atlas arm (spec `planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-atlas-arm.md`, Approved 2026-11-25)

Branch `feat/ct-pneumonitis-atlas`. `ATLAS=1` switches to `program_agentic_atlas/` (the same reader tools plus
`tools/atlas.py`), its own work folder (`~/.cache/alo-tasks/pneumonitis-variants-atlas`), and in place of the 10
similarity-picked side-by-sides a fixed atlas: exactly 10 marked examples per cause, all staged for every scan (never
one from its own patient), with an index of each entry's findings tied to slice numbers. Round 0 of the loop is the
literature review (`guide/literature.md`: at least 5 cited sources per cause; every guide pattern cites one); rounds
1-5 build and revise the atlas, the guide and the review. Launcher `scripts/run_atlas.sh` (starts only after the
variants study is done; compares atlas, guide arm, zero-shot and the probe on the same 134 VAL scans), cap $300.

| Build decision | Answer |
|---|---|
| Atlas images | `atlas.py draw`: up to 4 axial slices, 768 px, arrows and boxes, JPEG quality 85: ~90 KB on a real scan, ~4.5 MB for 50 (an RGB PNG was ~400 KB, a palette PNG ~360 KB) |
| Search | up to 80 requests every round (the spec's round-0 budget kept for all rounds, so the sealed setup does not change mid-run) |
| Builder session | 2 hours (10 write-ups per cause plus the review) |
| Atlas tool | a new `tools/atlas.py` in the atlas program, not an edit of `examples.py`, so the variants study's reader tools are unchanged |
| Guide-read count | `scripts/usage.py` also counts `cat`/`head` of a guide file (Read-only counting reported 59% of round-1 sessions; all 80 read it) |

## Library arm (spec `planning/vista_eval_vlm/2026-11-25-ct-pneumonitis-library.md`, Approved 2026-11-25)

Branch `feat/ct-pneumonitis-library`; supersedes the atlas arm (held in round 0). `LIBRARY=1` switches to
`program_agentic_library/` (the reader's tools plus `tools/library.py`, over an editable `library/`: wiki pages,
Stanford examples, catalogue) and its own work folder (`~/.cache/alo-tasks/pneumonitis-variants-library`).
Literature: review v2 (86 sources) and 252 images (144 open access, 108 from paywalled papers), the reader sees all.
Discovery (`scripts/run_loop.py discover`): director plan, five per-cause managers, director synthesis, each a sealed
Opus session that may fan out subagents, over all 599 TRAIN scans (`scripts/library_assets.py train`). Then a fixed
5-round climb (round 1 reads discovery's library; from round 2 the director revises first), then VAL once. Launcher
`scripts/run_library.sh`, cap $450 (discovery $175).

| Build decision | Answer |
|---|---|
| Discovery set | `build_data.py discovery-set`: all 599 eligible TRAIN scans as pools.json `train_all` (250 more converted); the climb's reader rounds keep rounds.json |
| Caches | the 599 TRAIN scans (~22 GB) and the literature images copied once to the work folder's disk; every session gets hard links (stage_examples.stage_library); the TRAIN cache is deleted after VAL |
| Own patient | examples from the reader's own patient are listed in `library_assets/exclude.json`, hidden by the tool, and their CTs not linked |
| Library use | free (no required opens); `library_log.jsonl` per session, summarised in the reader feedback |
| Climb order | round 1 reads discovery's library (no builder); rounds 2-5 director then reader, so the final version is one the reader used on TRAIN |
| Builder budget | $25 a session (fan-out), 2 hours; discovery stops before a session that could pass $175 |
| Merged score | "any drug" = max(other drug, immunotherapy), four classes, in scripts/compare.py beside the five causes |
