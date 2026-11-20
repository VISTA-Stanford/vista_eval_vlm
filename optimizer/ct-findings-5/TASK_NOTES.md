# ct-findings-5: task notes

Spec: `planning/vista_eval_vlm/2026-11-10-ct-agent-probability-climbs.md` on the mount (Approved 2026-11-10), the
second deliverable. Built 2026-11-10 from died-2yr-ct's probability mode (its scan staging, slice viewer, route,
stop rules, launcher and cost check), on branch `feat/ct-findings-agentic` off `feat/died-2yr-ct-agentic`. Engine
pin agentic-label-opt bb216c2 (`feat/vertex-login-plus-seat-pause`), Claude Code 2.1.307. Probability only: there
is no yes/no version of this task.

## The idea

One sealed Claude Code session per CT scan (Sonnet 5.5) answers the five CT findings questions the paper's Fig 5
asked every VLM, each as a 0-100 probability, and agentic-label-opt's optimizer (Opus 5.5) rewrites the
instructions and tools it follows, climbing on mean AUROC from a plain zero-shot start. The question: can the
agent's reading rank scans like the CT probes (Pillar0, COLIPRI, MerlinOnc) do?

## Decisions

| Decision | Answer |
|---|---|
| Questions | vista_bench v1.7 `per_ct_diagnostic_tasks`: `per_ct_lesion_liver`, `per_ct_lesion_lungs`, `per_ct_lesion_lymph_nodes`, `per_ct_lung_lobe_rul`, `per_ct_pneumonitis` (Phil 2026-11-10). Wording from vista_eval_vlm main `src/evalset/cards/lesions.py` |
| Answer | one session per scan: a whole number 0-100 per question (stored 0..1) plus 3-5 `reasons` (`finding`, `where`), never scored. One bad number fails the whole answer (scored 0.5 on all five) |
| Gold | the table's label per scan and task; -1 or no row = unscored for that question (not negative) |
| Score | AUROC per question over its scored scans (Mann-Whitney, ties half; a failed answer 0.5), mean over the five (Phil 2026-11-10). TRAIN breakdown per question: AUROC, Brier, mean probability for yes/no, calibration by tenth, distinct values, share on the most common, yes/no/unscored counts. VAL shows the optimizer the headline only |
| Mistake | a probability on the wrong side of the question's TRAIN-pool yes rate; only picks what the optimizer studies |
| Scans | one per patient, from the benchmark's train and valid splits only (TEST sealed; one early aggregate label count by split on 2026-11-10, nothing written, recorded in the spec). Volumes from `ct_data/ct_scans/vista/feb26/` (the TEST sample's folder); a drawn scan with no volume is dropped, counted and replaced; over 10% missing stops the build |
| Pools | VAL 200 (valid split), TRAIN 400 (train split), drawn by `scripts/build_data.py select` with seeds `ct-findings-5-v1:pool:val` / `:train`. Enrichment: in a seeded patient shuffle, take the next patient with a scan having the needed answer until every (question, yes/no) has 30 per 200 scans, scarcest first, then fill at random. Recorded in `pools.json` `_draw` |
| TEST split rule | none of the 527 patients of the VLMs' TEST sample (`vistabench/vlm/evalset_v1/samples/step2.parquet`) is in the pools or anywhere in the table's train/valid rows (build stops otherwise; `check_task.py` re-checks) |
| Seed | pure zero-shot (spec): the card's answer-key definitions, the five questions, ask for 0-100, and the starter slice viewer. No clinical hints, no tools from the death climb |
| Climb rules | as died-2yr-ct's probability mode: TRAIN `linear(100, 400, over=10)`, VAL `constant(200)`, 10 rounds, stop after 2 with no new best, $0.50/scan, $50/round, $200/run (run.py), 2 at a time, 1000 search requests/round, Opus 5.5 optimizer at effort high, $3 per optimizer session |
| Route | seat by default; `ALO_ROUTE=vertex` (the spec's route, from phil-sllm-02) with the after-run audit in the cost check |
| Work folder | `~/.cache/alo-tasks/ct-findings-5`, run `prob-r1`; results to `chaudhari_lab/phil/ct-findings-5-v1_7-runs/<host>/<run id>/` |
| Data | `chaudhari_lab/phil/ct-findings-5-v1_7/` (`splits/{train,val}.jsonl`, `pools.json`, `patients/<id>/`); id map (id -> person, accession, NIfTI) in `.../ct-findings-5-v1_7-private/`. Ids are salted hashes of the scan |

## Data (built 2026-11-10)

- VAL 200 of 5,569 valid scans (955 patients); TRAIN 400 of 27,257 train scans (4,379 patients). 0 drawn scans lacked a volume.
- VAL-200 yes / no / unscored: liver 50/137/13, lungs 146/47/7, lymph nodes 99/94/7, right upper lobe 43/87/70, pneumonitis 30/167/3.
- TRAIN-400: liver 103/263/34, lungs 284/95/21, lymph nodes 160/219/21, right upper lobe 87/170/143, pneumonitis 60/338/2.
- Volumes: died-2yr-ct's converter (HU int16, RAS, ~2.5 mm slabs, no header). About 13 s per scan per worker; the
  largest pool scan peaks at 4.8 GB, so 3 workers.

## Baselines on VAL-200 (`scripts/score_baselines.py score --ids pool`)

Linear probes from `vistabench/results/v1_7/ct/per_ct_diagnostic_tasks/`, joined by accession number; all cover
every scored scan; 0 published labels differ from the task's. `reproduce` matches every published VAL AUROC.

| Probe | liver | lungs | lymph nodes | RUL | pneumonitis | mean |
|---|---|---|---|---|---|---|
| Pillar0 | 0.711 | 0.824 | 0.845 | 0.893 | 0.815 | 0.818 |
| COLIPRI | 0.674 | 0.795 | 0.804 | 0.823 | 0.778 | 0.775 |
| MerlinOnc pan-cancer ep99 | 0.704 | 0.819 | 0.812 | 0.844 | 0.741 | 0.784 |

## Commands

    uv run --project $AGENTIC_LABEL_OPT --with scikit-learn --with pandas --with pyarrow python -B scripts/check_task.py check
    uv run --project $AGENTIC_LABEL_OPT python run.py dry-run --iterations 4       $0; ends STOPPED (no_new_best)
    uv run -q --no-project --with pandas --with pyarrow --with scikit-learn python -B scripts/score_baselines.py reproduce | score --ids pool
    uv run --project $AGENTIC_LABEL_OPT python run.py rescore --run <work folder>   $0: each VAL pass per question
    ALO_ROUTE=vertex bash scripts/cost_check.sh                                    PAID (~$5, Phil's go)
    ALO_ROUTE=vertex bash scripts/run_on_vm.sh                                     PAID climb (Phil's go)

## Hard cases and noise

The answer is the report as a language model read it, not the images: tiny "too small to characterize" liver
lesions count as yes; stable treated nodules may or may not be called benign; a third of reports do not let the
lobe be decided (unscored for RUL). Chest-only fields of view show only the top of the liver; non-contrast and
low-dose scans hide liver lesions.
