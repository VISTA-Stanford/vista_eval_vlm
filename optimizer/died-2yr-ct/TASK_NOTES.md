# died-2yr-ct: task notes

Set up with agentic-label-opt's `/new-labeling-task` (branch `fix/new-labeling-task-dogfood`), 2026-10-20;
engine pin 36dd87c on `feat/optimizer-websearch`. Every answer below is Phil's unless marked as a default.

**Since 2026-10-27** it is the CT arm of the agent in the eval-set spec
(`planning/vista_eval_vlm/2026-10-26-vlm-agent-eval-set.md`, work package 2), and these rows of the table below are
replaced: **Answer**: two forced yes/no answers in one session, death within 1 year and within 2 years of diagnosis;
anything but exactly yes or no, or a failed session, is wrong at both. **Gold**: `label` (2 years) and `label_1yr`
(`died_any_cause_1_yr`, same patient and scan, added by `scripts/build_data.py horizons`; old files in
`splits-2yr-only/`); TRAIN 718 died by 1 year and 470 in year 2, VAL pool 35 and 63. **Metric**: balanced accuracy
at each horizon, averaged. **Failed session** and **Calibration**: no longer apply (a failed or unreadable answer is
wrong at both horizons; there are no probabilities to calibrate). **Per-patient cap**: $0.50. **Optimizer web search**: off; the optimizer asks the
engine's search proxy, which refuses patient-specific questions and logs every request (Phil 2026-10-22).
**Route**: one setting for all sessions, the seat by default (engine `isolation.route`; token at
`~/.config/alo/oauth_token`). No Vertex-only service account can be made, so Vertex is future work (Phil 2026-10-28);
the code keeps it: `ALO_ROUTE=vertex` mounts only a Vertex-only key at `~/.config/alo/vertex-sa.json`.
Engine pin: agentic-label-opt main 9aee75f, shared with died-2yr-ehr; Claude Code 2.1.307. The finished run
`first` (probability, VAL AUROC 0.776) stays as history. Cost check: `bash scripts/cost_check.sh` (paid, ~$5,
Phil's go).

## The idea

A Claude Code session per patient predicts vista_bench v1.7 `died_any_cause_2_yr` from that patient's CT
alone, and agentic-label-opt's optimizer rewrites the instructions (and tools) the sessions follow. The point
is a head-to-head with the CT image models on the leaderboard (MerlinOnc v2.2 10-crop linear probe, test
AUROC 0.758 in v1.6). It is not strictly like-for-like, since the probe trains on about 3,500 labelled scans
and the agent learns only through instructions written from a few hundred; Phil accepts that as the best
comparison this kind of system allows. First of several vista_bench tasks.

## Decisions

| Decision | Answer |
|---|---|
| Input | the CT only (Phil: "just the CT scan, not a full OMOP export", for parity with the CT models) |
| Which CT | the one the label table pairs with the patient: latest qualifying chest CT on or before first treatment - 7 days (median 21 days before), the same scan the CT probes used |
| Gold | `died_any_cause_2_yr` as the table has it; -1 (too little follow-up) and no-split patients dropped. The 24 months count from diagnosis, not treatment (`get_progression_recurrence_survival_labels.sql:31`), although the question text says "treatment initiation": scored as is, so numbers compare with the leaderboard |
| Answer | a probability of death (Phil: fine) |
| Metric | AUROC, as the leaderboard scores the task |
| Failed session | scored at 0.5 and counted in the breakdown (Phil 2026-10-20) |
| Patient data | de-identified; Claude subscription route approved for VISTA data (Phil 2026-10-17) and for de-identified CT slice images (Phil 2026-10-20). `PATIENT_DATA = False` |
| Program mode | `agentic`: one sealed Claude Code session per patient |
| Per-patient model | Sonnet 5.5, `claude-sonnet-5-5` |
| What the session gets | the CT volume (`ct.npz` + `meta.json`) plus a starter slice viewer in `tools/`; the optimizer may rewrite it and add tools (Phil: "volume + tools") |
| Seed | only the single-CT VLM question (`prompts_by_task.single_ct_v1_7.json`, `per_ct_died_any_cause_2_yr::ct_current`), asking for a probability (Phil: "a short task description a la the VLM evals but nothing else") |
| Editable | `instructions.md` and anything under `tools/` |
| Optimizer | Opus 5.5 (`claude-opus-5-5`), effort high, $3 per session cap |
| Optimizer web search | on: WebSearch only, for literature and calibration anchors; page fetching off; no patient details in queries (Phil 2026-10-20). Engine branch `feat/optimizer-websearch` (36dd87c), live seal check 8/8 PASS |
| Calibration | AUROC still picks the best version; the brief asks for calibrated probabilities and Brier plus deaths per decile are reported (Phil 2026-10-20) |
| Segmentation | deferred: no precomputed organ masks, not pushed in the brief (Phil 2026-10-20) |
| Iterations | 5 (default) |
| Pools | fixed by `scripts/build_data.py`: TRAIN 400 (129 died), VAL 200 (63 died), natural death rate, seeded |
| Schedule | TRAIN `step(30, add=15, every=1, max=90)` (default), VAL `constant(200)` (Phil: 200, noise about +-0.035) |
| Spending ceiling | stop and ask before an iteration projected over $100 (Phil). Per-patient session cap $0.40 after the 10-patient cost check (seed $0.05, first version $0.10-0.15 a patient; optimizer $1.81) |
| Where it runs | not phil-sllm-01: its Cloud NAT gives each VM 64 ports and the first run stopped `proxy_died` 3 minutes in while another run shared them. `scripts/run_on_vm.sh` sets up another VM, runs 3 sessions at once, and resumes after a network stop (at most 20) |
| TEST | once, at the end, on all 852 TEST patients with a CT, and also on the subset MerlinOnc's probe scored, so the two compare on identical patients |
| Repo | vista_eval_vlm, `optimizer/died-2yr-ct/`, branch `feat/died-2yr-ct-agentic` |

## Data

- Built by `scripts/build_data.py` into `/mnt/su-vista-uscentral1/chaudhari_lab/phil/died-2yr-ct-v1_7/`
  (`splits/`, `pools.json`, `patients/<id>/`), id map in `.../died-2yr-ct-v1_7-private/`. On the mount
  because the VM's disk (66 GB free) cannot hold the volumes.
- Patients: TRAIN 3,886 (30.6% died; 10 had no NIfTI), VAL 811 (31.7%), TEST 852 (32.2%; 2 no NIfTI).
- Volumes: Hounsfield units, int16, reoriented to RAS (checked by eye on a rendered axial, coronal and
  sagittal slice), thin scans averaged into slabs of about 2.5 mm, no header carried over. About 4 s and
  roughly 20 MB per patient.

## Reference scores

MerlinOnc CT probe 0.758 (v1.6 test); radiology-report text 0.848; clinical features about 0.78; open
VLMs on CT images 0.52 to 0.61 (v1.1). Ceiling named in the standing instructions: 0.76.

## Hard cases and noise

Non-cancer deaths invisible on CT; long survivors with advanced disease on modern treatment; chest-only
fields of view that hide the liver and adrenals; non-contrast and low-dose scans.
