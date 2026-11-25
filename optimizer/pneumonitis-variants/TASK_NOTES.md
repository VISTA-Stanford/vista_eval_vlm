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
