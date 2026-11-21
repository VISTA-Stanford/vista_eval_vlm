# radprog-ct: task notes

Spec: `planning/vista_eval_vlm/2026-11-21-radprog-vlm-agent-arms.md` on the mount (Approved 2026-11-21). Copied
2026-11-21 from `optimizer/tb-questions` (the tumor board imaging agent, feat/tb-agent-questions 716bf60), imaging path
only, on branch `feat/radprog-vlm-agent` off main. Engine pin agentic-label-opt bb216c2, Claude Code 2.1.307.

## The idea

One sealed Claude Code session per scan pair (Sonnet 5.5) reads the CURRENT chest CT and the PRIOR CT its report was
read against, and gives the probability that the cancer progressed; agentic-label-opt's optimizer (Opus 5.5) rewrites
the instructions it follows, climbing on AUROC from a zero-shot start. Compared on the same fixed 1,000 TEST pairs with
Argon, Opus, Astra and MedGemma asked directly (eval-set step 3, `--which test1000`) and with the CT embedding probes
(`python -m evalset.compare`).

## Decisions

| Decision | Answer |
|---|---|
| Task | vista_bench v1.7 `radiology_progression_tasks`, task `radiology_progression` (v2 labels, 11-04); the prior = the table's own prior columns (the scan the report was read against) |
| Answer | two-way, following MSK-CHORD: progression or mixed = 1, stable or response = 0 (Phil 2026-10-26, re-confirmed 2026-11-21) |
| Input | both CT volumes and the days between them; no report, no EHR (Phil 2026-10-26, 2026-11-21) |
| Question text | the eval-set step-3 card word for word (`src/evalset/cards/progression.py`), answer line swapped for the probability line; same text for every VLM (Phil 2026-11-21); `check_task.py` compares them |
| Score | AUROC; a failed or unreadable answer counts 0.5 (no information) |
| Pools | TRAIN 400 (table train split), VAL 200 (table valid split), simple random draw at the natural rate, seed `radprog-ct-v1:pools`; only pairs with both NIfTIs on the mount and every comparison probe's score (Phil 2026-10-26, 2026-11-21) |
| TEST | the eval set's fixed 1,000 (`samples/step3_test1000.parquet`: whole step-3 TEST list, probe-scored pairs, random 1,000, seed 20261121). Built only on Phil's go (`build_data.py test`, then `volumes --split test`) |
| Schedule | VAL whole (200) every round, so every round is scored on the same pairs (build decision 2026-11-21; Phil's 10-26 answer had VAL grow 100 -> 200); TRAIN feedback 40 -> 100 over 4 rounds (Phil 2026-10-26) |
| Models, route | Sonnet 5.5 per pair, Opus 5.5 optimizer, Vertex global, optimizer web search off (Phil 2026-10-26, 2026-11-21) |
| Caps | $0.75/session, $60/round, $250/run; stop after 2 rounds with no new best or 10 rounds (spec) |
| Stop and come back | best VAL AUROC under 0.65, or more than 10% of sessions failing, before any TEST look (spec) |
| Machine | climb on phil-sllm-02 (spec); cost check and zero-shot on any box with Docker and the optimizer login |
| Disk flushing | the launcher syncs after its dry run and keeps a 5 s `sync` loop alive while the climb runs: the tumor board imaging climb's 30 s proxy timeouts on sllm-03 (2026-11-21) came from unflushed GBs |
| Work folder | `~/.cache/alo-tasks/radprog-ct`, run `r1`; results to `chaudhari_lab/phil/radprog-ct-v1_7-runs/<host>/<run id>/` |
| Data | `chaudhari_lab/phil/radprog-ct-v1_7/` (`splits/`, `pools.json`, `patients/<id>/{prior,current}/` + `pair.json`); id map in `radprog-ct-v1_7-private/`. Ids are salted hashes of the CT accession |

## Data (built 2026-11-21)

- TRAIN 400 pairs from 367 patients, 125 yes (31%); four-way stable / progression / response / mixed 229 / 115 / 46 / 10.
- VAL 200 pairs from 153 patients, 63 yes (32%).
- No patient in both pools; none of the 1,000 TEST pairs' 454 patients in either. No pair missing a NIfTI.

## Commands

    export AGENTIC_LABEL_OPT=~/code/agentic-label-opt/.claude/worktrees/mortality-engine
    uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow --with pyyaml python -B scripts/check_task.py check   $0
    uv run --project $AGENTIC_LABEL_OPT python run.py dry-run --iterations 4      $0; ends STOPPED (no_new_best)
    bash scripts/cost_check.sh                                                    PAID 5+5 (approved)
    bash scripts/run_on_vm.sh                                                     PAID zero-shot + climb (approved, sllm-02)
    uv run --project $AGENTIC_LABEL_OPT python run.py test --tag program-vN       PAID TEST (needs Phil's go); writes answers_by_case.csv for evalset.compare

## Reference points (in the brief)

CT probes, current + prior, two-way, whole valid split: Pillar0 0.79, COLIPRI 0.78, MerlinOnc v2.2 0.75 (current scan
alone 0.005-0.007 lower). On the 1,000 TEST pairs: 0.815 / 0.764 / 0.752.
