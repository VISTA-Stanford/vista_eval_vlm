Reference: research-skills/claude_ops.md

<!-- Authored by Codex (gpt-5.5) via `codex exec -s read-only` on 2026-07-15, auditing committed
     diff 1b24507 (the in-session OOM load-path fix). The read-only sandbox could not write files,
     so the Codex section below is its stdout transcribed verbatim (Claude was scribe, not reviewer).
     This is a SEPARATE review from `...-implementation-feedback.md` (the earlier Axis A/B review). -->

# Implementation Feedback: CT load-path OOM fix (commit 1b24507) — vlm-ct feb26 v1_5 rebaseline

## Verdict
Revise. The lazy `dataobj` slicing is behavior-preserving for NIfTI scaling/dtype semantics and fixes the OOM class, but the new mount route should fail over to the storage-client download on mount read failures before this shared inference path is treated as robust.

## Behavior-Preservation
The `get_fdata()` → `dataobj[:, :, index]` change is semantically sound for the loaded NIfTI slices. Old code loaded the full volume with `img_obj.get_fdata()` and then sliced `img_data[:, :, index]`; new code loads a proxy image and slices `ct_img.dataobj[:, :, index]`, then explicitly casts to `np.float64` at each slice site: `src/vqa_dataset.py:200`, `src/vqa_dataset.py:237`, `src/vqa_dataset.py:259`, `src/vqa_dataset.py:280`. Nibabel `ArrayProxy` applies header slope/intercept scaling when materialized, and `get_fdata()` defaults to float64, so `np.asarray(..., dtype=np.float64)` matches the old dtype and scaling for the selected planes.

Shape/depth selection is also preserved: old code used `len(img_data.shape) > 2` and `img_data.shape[2]`; new code uses `len(ct_img.shape) > 2` and `ct_img.shape[2]`, which comes from the image header/proxy shape: `src/vqa_dataset.py:223`, `src/vqa_dataset.py:245`, `src/vqa_dataset.py:266`. For 4D or otherwise unusual images, both old and new forms slice with `[:, :, index]`, so any downstream oddity is not introduced by this commit.

The main intentional behavioral difference is memory/cache behavior: the old path filled a full-volume float64 array; the new path performs repeated lazy slice reads and never materializes the whole CT. The VM evidence reports byte-identical `image_hashes` on `--limit 3`, but the stronger reason this generalizes is the proxy scaling + explicit float64 cast at every slice extraction site.

## Correctness / Safety (shared inference load path)
- Medium | Mount read failures do not fall back to GCS download. | Evidence: source priority selects `use_mount` before storage-client download at `src/vqa_dataset.py:184` and `src/vqa_dataset.py:192`; if `nib.load(str(mount_path))` or a later lazy slice read fails, the broad `except` at `src/vqa_dataset.py:286` logs and continues text-only, bypassing the available download route at `src/vqa_dataset.py:208`. Required fix: if `use_mount` fails and `self.storage_client is not None`, retry the same blob through the temp-file download path before failing closed.

- Low | Mount acceptance is only `Path.exists()`, so a stale/corrupt but syntactically valid file at `/mnt/<bucket>/<blob>` can silently become authoritative. | Evidence: `mount_path = Path("/mnt") / self.bucket_name / blob_path` and `use_mount = (not use_local) and mount_path.exists()` at `src/vqa_dataset.py:184`-`src/vqa_dataset.py:185`; the code does not check `is_file()`, size, generation, or retry remote bytes. Required fix: at least require `mount_path.is_file()` and add the fallback above; if generation integrity matters, compare against GCS metadata before trusting the mount.

- Low | `indexed_gzip` reaches the recommended setup path, but not the project dependency/lock graph. | Evidence: `requirements-default.txt:72` pins `indexed_gzip==1.10.3`; recommended setup installs that file at `README.md:21` and `scripts/setup.sh:39`-`scripts/setup.sh:40`. But `pyproject.toml:7`-`pyproject.toml:12` lists `nibabel` without `indexed_gzip`, and `uv.lock:477`-`uv.lock:499` / `uv.lock:669`-`uv.lock:678` likewise omit it. Without `indexed_gzip`, `.nii.gz` proxy slicing should remain correct but can become very slow because gzip seeking is inefficient; this is a performance/runtime risk, not an expected pixel correctness change. Required fix: either add `indexed_gzip` to `pyproject.toml` and refresh `uv.lock`, or explicitly declare that plain `uv sync` is unsupported for CT inference.

- Correct | Fail-closed null/missing CT behavior is preserved. | Evidence: `resolve_ct_blob` returns `None` for null/NaN/empty UIDs at `src/vqa_dataset.py:43`-`src/vqa_dataset.py:50`; `__getitem__` only enters the CT load block when `resolved is not None` at `src/vqa_dataset.py:172`-`src/vqa_dataset.py:173`; otherwise `img` remains `None` from `src/vqa_dataset.py:119`.

- Correct | Temp-file lifecycle is safe. | Evidence: `tmp_file_path = None` is initialized before the `try` at `src/vqa_dataset.py:175`; only the GCS-download route creates it at `src/vqa_dataset.py:209`-`src/vqa_dataset.py:212`; cleanup runs in `finally` after slice processing at `src/vqa_dataset.py:289`-`src/vqa_dataset.py:294`. No unlink happens before lazy reads complete, and the guard avoids unlinking `None`.

- Correct | Exception handling still catches load, download, proxy slice, and image-processing failures, then continues text-only. | Evidence: the broad `try` starts before `nib.load` and route selection at `src/vqa_dataset.py:176`; lazy slice reads are inside the same block at `src/vqa_dataset.py:237`, `src/vqa_dataset.py:259`, and `src/vqa_dataset.py:280`; the `except` logs and leaves `img` as `None` at `src/vqa_dataset.py:286`-`src/vqa_dataset.py:288`.

- Watch | Real inference has more concurrency than the golden harness. | Evidence: `golden_harness.py` iterates `PromptDataset` directly at `src/vista_run/golden_harness.py:213`; real `run_bq` uses `DataLoader(num_workers=4, prefetch_factor=8, persistent_workers=True)` at `src/vista_run/run_bq.py:950`-`src/vista_run/run_bq.py:959`. This increases concurrent gcsfuse random reads, so the mount fallback issue matters outside the weight-free harness.

## Defensible Deviations
- Reading from `/mnt/<bucket>/<blob>` before downloading is a defensible OOM/performance improvement when the mount is healthy: `src/vqa_dataset.py:180`-`src/vqa_dataset.py:185`.
- Moving temp-file creation inside only the GCS-download route is an improvement: local and mount reads no longer create unnecessary files, and cleanup is centralized: `src/vqa_dataset.py:208`-`src/vqa_dataset.py:214`, `src/vqa_dataset.py:289`-`src/vqa_dataset.py:294`.
- Changing the CT source label from binary `local|gcs|none` to `local|mount|gcs|none` improves observability: `src/vqa_dataset.py:190`-`src/vqa_dataset.py:198`.

## Suggested Code Edits
- `src/vqa_dataset.py:204`-`src/vqa_dataset.py:212`: wrap mount `nib.load` plus lazy slice processing so mount failures retry via `download_to_filename` when `self.storage_client` exists, then only fail closed if both routes fail.
- `src/vqa_dataset.py:185`: change mount gating from `mount_path.exists()` to at least `mount_path.is_file()`, with optional size/metadata validation if stale mount cache is a known risk.
- `pyproject.toml:7`-`pyproject.toml:12`: add `indexed_gzip` to project dependencies and refresh `uv.lock`, unless the team explicitly does not support `uv sync` for inference/CT runs.

## Questions For The Author
- Is `/mnt/su-vista-uscentral1` guaranteed in production inference to be a coherent gcsfuse mount of the same bucket, or should storage-client bytes remain the integrity fallback whenever available?
- Should plain `uv sync` be a supported environment path for CT inference, or is `scripts/setup.sh` the only supported provisioning route?

## Audit Trail
- docs/plans/vlm-ct-feb26-v1_5-golden-rebaseline.md
- docs/vm-status/2026-07-14-rungs1-2-v1_5-stage-and-prebank.md
- src/vqa_dataset.py
- requirements-default.txt
- pyproject.toml
- uv.lock
- README.md
- scripts/setup.sh
- configs/all_tasks.yaml
- src/vista_run/run_bq.py
- src/vista_run/run.py
- src/vista_run/golden_harness.py

---

## Claude's classification (2026-07-15, planner Mac) — how the findings were adjudicated

Behavior-preservation is the gate's real purpose, and it is **CLEAN** — C1 (banked at `1b24507`) is safe to keep. No correctness finding survived. The three "Revise" findings are all robustness/packaging on the **shared loader**, and each is deferred rather than applied inline, for two reasons:

1. **Loader-freeze for the 3b byte-identity net.** C2/3b diffs C1 (before) vs C3 (after) with the resolver **and NIfTI loader held constant** (plan Axis C-3). Editing the loader (mount→GCS fallback, `is_file()`) between C1 and C3 would put a different loader on each golden side, risking a spurious diff. Loader hardening therefore lands **after** the 3b diff closes (then optionally re-bank C1 on the hardened loader).
2. **Machine posture.** Applying + verifying a shared-loader change is executor-VM work (needs a re-run / re-bank); the planner Mac cannot verify it. So it is a VM-bound follow-up regardless of when it's sequenced.

- **Medium (mount fallback):** AGREE it's a real robustness gap (confirmed by the "Watch" DataLoader concurrency note — matters for real inference). DEFER to the post-3b loader-hardening follow-up.
- **Low (`is_file()`):** AGREE, trivial tightening. Bundle with the same follow-up.
- **Low (`indexed_gzip` not in pyproject/uv.lock):** Partly moot — the VM's in-lane provisioning correction already established plain `uv sync` is **not** this repo's supported path (`setup.sh` installs `requirements-default.txt`, which carries `indexed_gzip`), so it **does** reach the runtime. Remaining nicety = add to `pyproject.toml` + refresh `uv.lock` (can't refresh the lock on the Mac), or document `uv sync` as unsupported for CT. → packaging follow-up. **This answers Codex's Question 2.**
- **Question 1 (is `/mnt` coherent in prod?):** genuine — drives whether the storage-client bytes should stay the integrity fallback. Folded into the follow-up's design.
