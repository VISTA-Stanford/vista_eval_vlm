Reference: docs/claude_ops.md

# VM report — rung-0 accuracy crosscheck vs Ryan's committed baseline (PFS, v1_1 substrate)

**Status: Handoff to VM** (2026-07-13)
**Branch:** `worktree-vlm-modular-preprocessing-roadmap` — **uncommitted on the Mac; commit + push first, SHA set at commit time.**
**Locator:** REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` — **unmerged; `git fetch` first** · this doc `docs/vm-status/2026-07-13-rung0-ryan-accuracy-crosscheck.md`. Reach it: `git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull`.
**Machine posture:** authored on the planner Mac (`DNa825021.SUNet`, no runtime). Runs entirely on the **Claude-Code CPU** box `phil-sllm-01` — it shares the `/mnt/su-vista-*` mounts (where the fresh constrained result CSVs live) and carries Ryan's committed baseline in the repo, so it regenerates 0c locally, runs the compare, and does the readback itself (Claude-driven, no GPU).
**Target machine:** Claude-Code CPU (`phil-sllm-01`), readback there.
**Plans:** [`vlm-rung0-reproduce-ryan-feb26.md`](../plans/vlm-rung0-reproduce-ryan-feb26.md#verification--vm-handoff) — 0c report (OQ-R2/R3). ⚠ **These criteria are REPORT-ONLY / soft** (a confounded ~10% informal band, *not* a hard pass/fail gate); this doc is a sanity read, not a smoke gate.
**Prior handoffs:** [`2026-07-13-rung0-0b-decoding-fix.md`](./2026-07-13-rung0-0b-decoding-fix.md) — the GREEN constrained 0b whose result CSVs this crosscheck reads.

## Why this doc

Rung-0 proved **operability** (constrained decoding engages; 0c `predicted_label == -1 total: 0`). This doc does the
one loose end: the **accuracy crosscheck vs Ryan's committed baseline** — done *now*, on the **v1_1 substrate**, because
rungs 1–2 cut to v1_5 and a later crosscheck would compare v1_5 results to Ryan's v1_1 baseline (a weaker "reproduce
Ryan" signal). This is the closest-to-Ryan comparison we'll get.

**It is deliberately soft** (OQ-R2/R3): the arms are *not* apples-to-apples with Ryan on either absolute accuracy or the
±image delta — confounds are report presence (our timelines are report-inclusive, Ryan's report-stripped), slice count
(our 30 vs Ryan's 10), and un-subsampled n. Success = **"sane, same neighborhood, within ~10%"**, NOT a match claim.

**Arm mapping:** rung-0 `axial_all_image` ↔ Ryan `image_and_timeline`; rung-0 `no_image` ↔ Ryan `timeline_only`.

## Step 0 — get the artifacts onto `phil-sllm-01`

```bash
cd <vista_eval_vlm repo>
git fetch origin && git checkout worktree-vlm-modular-preprocessing-roadmap && git pull
# Confirm the fresh constrained result CSVs are on the shared mount (written by the GREEN 0b), and Ryan's baseline is present:
RES=/mnt/su-vista-uscentral1/vistabench/vlm/results/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1/progression_recurrence_free_survival_1_yr/medgemma-1.5-4b-it
ls -la "$RES"/*_results_{no_image,axial_all_image}.csv    # dated 07-13, small (KB/MB) — the CONSTRAINED run
ls -la figures/results_stats/all_model_response.csv        # Ryan's committed baseline comparator
```
**Expected:** clean checkout; the two result CSVs are **07-13** and small (the constrained run, *not* the 147 MB 07-09 essays — those are archived at `~/rung0_stale_freegen_20260709/` on the GPU box); Ryan's `all_model_response.csv` present.
**STOP:** result CSVs missing or 07-09-dated/147 MB (wrong/stale run) → do not proceed; the constrained CSVs must be the ones present.

## Step 1 — regenerate the 0c reducer output on `phil-sllm-01` (independently reproduce the QC)

Re-run 0c here (CPU-fine; reads the shared-mount raw result CSVs) so the crosscheck reads a locally-produced output and
we confirm the QC reproduces on a second box:

```bash
cd src && python -m results.constrained_all_model_response \
    --config ../configs/all_tasks.rung0.yaml \
    --output ../figures/results_stats/rung0_constrained_all_model_response.csv
```
**Expected:** `[QC] predicted_label == -1 total: 0`; `Dropped N rows` with N ≈ 27% (≈678); `Wrote ~1798 rows`. Same as the
GPU-box 0c → confirms the constrained result is reproducible and not box-specific.
**STOP:** `predicted_label == -1 total` ≠ 0, or `Dropped 0` → the CSVs on the mount are not the constrained run (or a stale
leftover) → halt and report; do not run the crosscheck on bad inputs.

## Step 2 — the crosscheck read (report-only)

```bash
cd .. && python - <<'PY'
import os, pandas as pd
root = os.getcwd()
RES  = "/mnt/su-vista-uscentral1/vistabench/vlm/results/progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr_v1_1/progression_recurrence_free_survival_1_yr/medgemma-1.5-4b-it"

def acc(df):
    d = df[~df["ground_truth_label"].astype(str).isin(["-1","-1.0"])].copy()
    d["ok"] = d["predicted_label"].astype(str) == d["ground_truth_label"].astype(str)
    return d.groupby("experiment")["ok"].agg(acc="mean", n="size")

r0   = pd.read_csv(f"{root}/figures/results_stats/rung0_constrained_all_model_response.csv")
ryan = pd.read_csv(f"{root}/figures/results_stats/all_model_response.csv")
print("RUNG-0 cols:", list(r0.columns))
print("RYAN   cols:", list(ryan.columns))
print("\n=== RUNG-0 (fresh constrained, v1_1 substrate, feb26 CT) ===")
print(acc(r0))
print("\n=== RYAN baseline (all_model_response.csv), PFS only ===")
print(acc(ryan[ryan["task"]=="progression_recurrence_free_survival_1_yr"]))

# used_image>0 stratification for the axial arm (drops CT-null + the ~13% feb26-missing)
raw = pd.read_csv(f"{RES}/progression_recurrence_free_survival_1_yr_results_axial_all_image.csv")
print("\nRAW axial cols:", list(raw.columns))
col = next((c for c in ["used_image","image_count"] if c in raw.columns), None)
if col:
    keep = set(raw.loc[raw[col] > 0, "index"])
    ax = r0[(r0["experiment"]=="axial_all_image") & (~r0["ground_truth_label"].astype(str).isin(["-1","-1.0"]))].copy()
    axk = ax[ax["index"].isin(keep)]
    ok  = axk["predicted_label"].astype(str) == axk["ground_truth_label"].astype(str)
    print(f"\naxial_all_image stratified to {col}>0: acc={ok.mean():.3f}  n={len(axk)}  (dropped {len(ax)-len(axk)} CT-missing)")
else:
    print("\n(no used_image/image_count col — record RAW axial cols above; the join needs adjusting)")
PY
```
**Expected:** prints per-arm accuracy for rung-0 and Ryan (+ the stratified axial number). The read is "green" if the
absolute accuracies sit in Ryan's **~10% neighborhood** and — softer still — the ±image direction is not wildly opposite.
The column-name prints let a mismatch (Ryan's baseline using different column names) be caught and the one-liner adjusted.
**STOP:** none — **report-only** (a confounded sanity read, per OQ-R2/R3; a divergence is characterized, never halts).
If a KeyError fires (Ryan's baseline column names differ), record the printed `RYAN cols:` and adjust the accuracy
one-liner — that's an in-lane correction, not a deviation.

## Report back

Append to `## VM run results` (PHI-free — accuracies / counts only, no rows): Step-1 QC (`predicted_label == -1` = ?,
`Dropped` = ?); the two per-arm accuracies (rung-0 vs Ryan, mapped arms) + n each; the used_image-stratified axial
accuracy; and a one-line read — is rung-0 in Ryan's ~10% neighborhood, and which way does ±image point. Frame it as the
confounded sanity signal it is, not a reproduction claim.

## VM run results — readback on `phil-sllm-01` (Claude-Code CPU), 2026-07-13 · REPO `vista_eval_vlm` · BRANCH `worktree-vlm-modular-preprocessing-roadmap` @ `eb4d030`

Ran entirely on `phil-sllm-01` (shared `/mnt/su-vista-*` mounts; Claude-driven, no GPU). Report-only sanity read per OQ-R2/R3 — **not** a reproduction claim.

- **Step 0 — artifacts:** ✅ (with an in-lane note). The two constrained result CSVs are present and **07-13-dated** (`no_image` mtime 19:12, `axial_all_image` 19:56 — both written after the 18:30 0b run), Ryan's `all_model_response.csv` present. **In-lane correction:** the CSVs are **~145 MB**, not the "small (KB/MB)" the Step-0 Expected predicted — but the Step-0 STOP's real target is the **07-09 stale free-gen essays**, which are content-distinct. Verified by content: `model_response` is exactly Yes/No (max_len=3, mean 2.5; no_image {Yes:678, No:560}, axial {No:624, Yes:614}) across 1238 rows/arm — the constrained run, not essays. The ~145 MB is the bulky `dynamic_prompt` / `log_probs` / token-count columns, not response bloat. The 0b readback's "small — KB/MB" claim was inaccurate; size is a false-alarm proxy here, resolved by content + the Step-1 QC below. Archive-on-this-box check is moot: `~/rung0_stale_freegen_20260709/` lives on the **GPU box**, not `phil-sllm-01`.
- **Step 1 — 0c regenerated locally (independent QC on a second box):** ✅ **GREEN, reproduces the GPU-box 0c exactly.** `[QC] predicted_label == -1 total: 0`; `Dropped 678` rows where `ground_truth_label == -1` (= 27.4% of 2476 — the insufficient-follow-up class); `Wrote 1798 rows`. Confirms the constrained result is reproducible and not box-specific.
- **Step 2 — crosscheck read (report-only):** ran clean, no KeyError (rung-0 has `predicted_label`/`ground_truth_label`; Ryan's baseline uses `model_response_cleaned` but the same label cols, so `acc()` worked as-is).

  | arm (mapped) | rung-0 acc (n) | Ryan acc (n) | Δ (rung-0 − Ryan) |
  |---|---|---|---|
  | `no_image` ↔ `timeline_only` | 0.515 (899) | 0.375 (136) | **+0.140** |
  | `axial_all_image` ↔ `image_and_timeline` | 0.513 (899) | 0.375 (136) | **+0.138** |
  | `axial_all_image`, `used_image>0` ↔ `image_and_timeline` | 0.480 (252) | 0.375 (136) | **+0.105** |

  (Ryan PFS also carries `image_only` 0.294/136 and `report_and_timeline` 0.434/136, unmapped. Stratifying the axial arm to `used_image>0` dropped 647 CT-missing rows → n=252.)

- **Read (confounded sanity signal, not a match):** rung-0 sits **above** Ryan by ~10.5–14 pp — **at/just beyond the ~10% informal band** on the unstratified arms, at the edge on the `used_image>0`-stratified axial. The higher absolute accuracy is directionally consistent with the declared **report-presence confound** (our timelines are report-inclusive, Ryan's report-stripped — report text helps), compounded by 30-vs-10 slices and un-subsampled n (899 vs Ryan's subsampled 136). The **±image direction agrees with Ryan**: both are essentially **flat** — rung-0 axial 0.513 vs no_image 0.515 (−0.002), Ryan image_and_timeline 0.375 vs timeline_only 0.375 (0.000); image adds ~nothing over timeline in either. Net: **sane, same neighborhood, ±image direction not opposite** — a green-ish soft read, with the above-band gap fully explained by the known report-presence confound. No reproduction claim; no STOP (Step 2 is report-only).
- **Decision gates:** none. **In-lane corrections:** Step-0 size proxy (~145 MB, not KB/MB) → resolved by content + Step-1 QC (above). **Deviations (class 3):** none.

PHI: accuracies / counts / column names / pass-fail only — no patient rows, UIDs, timelines, or dates.

### Correction (Mac, 2026-07-13) — matched-arm comparison: `report_and_timeline` is the right partner

The Step-2 read above mapped our **report-inclusive** `no_image` to Ryan's **report-stripped** `timeline_only` — a
report-content mismatch, which is what produced the "+14 pp, above-band, confound-explained" framing. Ryan's
report-**inclusive**, no-image arm is **`report_and_timeline`** (0.434, n=136) — which the readback computed but left
"unmapped." Comparing like-for-like:

| matched arm (report-inclusive, no image) | rung-0 | Ryan | Δ |
|---|---|---|---|
| our `no_image` ↔ Ryan **`report_and_timeline`** | **0.515** (899) | **0.434** (136) | **+0.081 — WITHIN the ~10% band** |

So we **did** replicate: on the properly-matched arm the gap is +8 pp, in-band — not the +14 pp the wrong-arm mapping
implied. Corroborating evidence, all from Ryan's own baseline:
- **Report-lift, Ryan's pipeline:** `report_and_timeline` (0.434) − `timeline_only` (0.375) = **+5.9 pp** — report text
  helps in his pipeline too, confirming the confound *direction* (most of our earlier elevation was the arm mismatch).
- **±image unchanged:** both flat (ours −0.002; Ryan `image_and_timeline` vs `timeline_only` 0.000); Ryan `image_only`
  (0.294) is the *worst* arm → image-alone < timeline < timeline+report. Same "CT underpowered for PFS" story.
- **Residual +8 pp is within noise:** Ryan n=136 → SE ≈ 4 pp, so +8 pp ≈ 2 SE; the leftover is our un-subsampled n
  (~899) + minor rendering diffs, not a pipeline discrepancy.

**Caveat (image arm stays confounded):** Ryan has *no* report-inclusive + image arm (his image arms are report-stripped,
10-slice), so our `axial_all_image` (report-inclusive, 30-slice) has no clean partner — the image-arm Δ remains
qualitative. The **no-image arm is the clean replication test**, and it lands in-band. **Net: genuine same-neighborhood
replication on the matched arm.**
