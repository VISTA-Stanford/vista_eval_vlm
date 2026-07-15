Reference: research-skills/claude_ops.md

<!-- Authored by Codex (gpt-5.5) via `codex exec -s read-only` on 2026-07-15, auditing the
     UNCOMMITTED 3b (C2) CT-adapter dissolution (src/vqa_dataset.py + src/context/adapters/ct.py).
     Read-only sandbox → Codex's stdout transcribed verbatim (Claude was scribe, not reviewer).
     No repo-local implementation-review checklist exists; a repo-grounded checklist would sharpen future reviews. -->

# Implementation Feedback: 3b CT-adapter dissolution (uncommitted) — vlm-ct feb26 v1_5 rebaseline

## Verdict
Ready to commit. The refactor keeps resolver/load in `PromptDataset`, moves only CT slice selection + windowing into `CTAdapter`, and I found no byte-identity or OOM/laziness regressions in the audited code.

## Plan Coverage

| Slice / section | Status | Evidence: path:line | Notes |
|---|---:|---|---|
| C2 boundary: adapter only select+window; resolver/load stay in caller | Done | `src/vqa_dataset.py:164`, `src/vqa_dataset.py:203`, `src/vqa_dataset.py:229`, `src/context/adapters/ct.py:64` | Caller still resolves `(study, series)` and loads NIfTI before passing `ct_img.dataobj`; adapter only consumes a passed volume. |
| Legacy CT experiment slice counts | Done | `src/vqa_dataset.py:61`, `src/context/presets.py:17,82,106,127` | Map is 30 for `axial_all_image` + both retrieved-with-image variants, 50 for `no_timeline`/`all_vb_image_only`, 10 for `no_report`; no extra experiments mapped. |
| Non-CT experiments remain no-image / non-CT path | Done | `src/vqa_dataset.py:102,129,215` | `_ct_slice_count=None` prevents CT image production for unmapped experiments even if UIDs exist. |
| Selector parity | Done | `src/context/selectors/ct_selectors.py:16,30`; legacy: `src/vqa_dataset.py` HEAD 227-237/249-260/270-281 | `evenly_spaced_k` reproduces `position=i/(k-1)`, `index=int(position*(depth-1))`, and the legacy `index >= depth` clamp, including `k == 1`. |
| Float64 slice parity | Done | `src/context/adapters/ct.py:79`; legacy: `src/vqa_dataset.py` HEAD 237/259/280 | New: `np.asarray(volume[:, :, idx], dtype=np.float64)` after `volume=ct_img.dataobj`; legacy: `np.asarray(ct_img.dataobj[:, :, index], dtype=np.float64)`. Same proxy slice + dtype. |
| Windowing byte parity | Done | `src/context/adapters/ct.py:48`, `src/context/windowing.py:19,29,39`; legacy: `src/vqa_dataset.py` HEAD 91-104 | Gemma path remains `multi_window_rgb -> round -> uint8 -> RGB -> pad`; non-Gemma remains `grayscale -> L -> pad`. |
| Caps parity | Done | `src/vqa_dataset.py:92,96`, `src/context/adapters/base.py:58,61,62`, `src/context/adapters/ct.py:62` | With empty adapter preprocess config, `resolve_model_caps` matches legacy `self.is_gemma` + `self.target_size`: Gemma 448 + RGB windowing, else 512 + grayscale. |
| selected_indices payload mapping | Done | `src/context/adapters/ct.py:85,88`, `src/vqa_dataset.py:232,236`; legacy: `src/vqa_dataset.py` HEAD 239-242/261-264/282-285 | Adapter may return `selected_indices=[]` for non-3D, but caller keeps legacy external value `None` when `img is None`. |
| Temp-file lifecycle | Done | `src/vqa_dataset.py:208,229,242` | Adapter contextualization runs inside the `try`; `finally` unlinks only after all lazy proxy slices are read. |
| Pickle safety | Done | `src/vqa_dataset.py:102,225`, `src/context/selectors/ct_selectors.py:107` | Dataset pickles with only `_ct_slice_count` int and `_ct_adapter=None`; closure from `resolve_ct_selector` is created lazily in the worker. |

## Byte-Identity Risks
- Empty.

## Correctness / Safety
- Empty.

## Defensible Deviations
- `CTAdapter` receives `ct_img.dataobj` instead of `ct_img`, while legacy guarded on `len(ct_img.shape) > 2`; with nibabel `>=5.3.3` declared in `pyproject.toml:11`, `ArrayProxy` shape/ndim access is expected to support the new `getattr(volume, "ndim", 0) > 2` guard at `src/context/adapters/ct.py:69` without materializing the volume.
- Adapter metadata records `selected_indices=[]` for non-3D/no-payload cases at `src/context/adapters/ct.py:88`; the public dataset output intentionally preserves legacy `None` via `src/vqa_dataset.py:236`.

## Suggested Code Edits
- Optional: update the stale comment at `src/vqa_dataset.py:128` from "load 50 axial slices like axial_all_image" to "load 30 axial slices like axial_all_image." The code is correct at `src/vqa_dataset.py:61`, but the comment contradicts it.

## Questions For The Author
- Empty.

## Audit Trail
- `docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md`
- `src/vqa_dataset.py`
- `src/context/adapters/ct.py`
- `src/context/adapters/base.py`
- `src/context/selectors/ct_selectors.py`
- `src/context/windowing.py`
- `src/context/block.py`
- `src/context/presets.py`
- `src/context/adapters/__init__.py`
- `src/context/adapters/ehr.py`
- `src/context/adapters/pathology.py`
- `src/vista_run/golden_harness.py`
- `src/vista_run/run_bq.py`
- `pyproject.toml`

---

## Claude's classification (2026-07-15, planner Mac)

Codex's verdict is **Ready to commit** with **zero** byte-identity or correctness/safety findings — every parity dimension (selector math, float64 slice, windowing, caps, index mapping, temp-file lifecycle, pickle) verified Done. This is the independent confirmation of the by-construction reasoning; the empirical net is still the C4 golden diff (Doc 2).

- **Suggested Code Edit (stale comment 50→30, `vqa_dataset.py:128`):** AGREE — the comment contradicted `_CT_SLICE_COUNTS` (axial_all_image + retrieved-with-image = 30). Pre-existing (not introduced by 3b), but trivially correct. **APPLIED.**
- **Defensible Deviation 1 (`ct_img.dataobj` vs `ct_img`):** CONFIRMED correct — this is the intended design (lazy ArrayProxy preserves the OOM fix); nibabel ≥5.3.3 (lock 5.3.3) exposes `ArrayProxy.ndim`/`.shape`. No change.
- **Defensible Deviation 2 (adapter `selected_indices=[]` vs caller `None`):** CONFIRMED correct — the caller maps payload-None → `selected_indices=None`, byte-matching the legacy no-image path. No change.
- **Byte-Identity Risks / Correctness-Safety / Questions:** none.

Applied: 1 Codex finding (comment). Not applied: 0. Deviations confirmed: 2. Open questions: 0.
