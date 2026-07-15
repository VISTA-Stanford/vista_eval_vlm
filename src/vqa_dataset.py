import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from pathlib import Path
import io
import tempfile
from vista_run.utils.utils_inference import pad_to_512, pad_to_size
# CT windowing/normalization now live in context.windowing (single home; the CT
# adapter shares them). multi_window_rgb == legacy `window`, grayscale == normalize_slice.
from context.windowing import multi_window_rgb, grayscale

# Go-forward CT snapshot. `nov25` is deleted from the bucket (zero objects); `feb26` is the
# only snapshot. The prefix is a config value (`paths.ct_snapshot_prefix`, threaded via
# PromptDataset(ct_snapshot_prefix=...)) so a future re-materialization is a config/data change,
# not a code edit — the modular-preprocessing thesis.
DEFAULT_CT_SNAPSHOT_PREFIX = "chaudhari_lab/ct_data/ct_scans/vista/feb26"


def resolve_ct_blob(study_uid, series_uid, prefix=DEFAULT_CT_SNAPSHOT_PREFIX):
    """
    Resolve a v1_5 CT (image_study_uid, image_series_uid) pair to its materialized blob.

    v1_5 drops the string `nifti_path` (now a deprecated INTEGER placeholder) and links CTs
    only by (study, series) UID: the blob layout is `{prefix}/{study}__{series}.nii.gz` in
    bucket `su-vista-uscentral1`. `prefix` is a config value (`paths.ct_snapshot_prefix`,
    default feb26) so a future re-materialization is a config/data change, not a code edit —
    the modular-preprocessing thesis. Single shared helper (do NOT inline the `{study}__{series}`
    assembly): `__getitem__`'s load path, the reprocessing-coverage report, and any future
    snapshot bump all call this, so the layout can't drift.

    Args:
        study_uid: image_study_uid value (string; null/NaN -> no CT).
        series_uid: image_series_uid value (string; null/NaN -> no CT).
        prefix: CT snapshot prefix (bucket-relative; default feb26).

    Returns:
        (blob_path, filename), or None if either UID is null/empty — the caller then treats
        the row as no-image (the same fail-closed result as a missing CT), and never reaches
        for the deprecated `nifti_path`.
    """
    if study_uid is None or series_uid is None:
        return None
    if pd.isna(study_uid) or pd.isna(series_uid):
        return None
    study = str(study_uid).strip()
    series = str(series_uid).strip()
    if not study or not series:
        return None
    filename = f"{study}__{series}.nii.gz"
    return f"{prefix}/{filename}", filename

class PromptDataset(Dataset):
    def __init__(self, df, prompt_col='dynamic_prompt', add_options=False, experiment='no_image', storage_client=None, model_type=None, ct_dir=None, ct_snapshot_prefix=None):
        """
        Args:
            df: Dataframe containing the data.
            prompt_col: The column name to use for the text prompt.
            add_options: Whether to append options to the prompt.
            experiment: Experiment type - 'no_image', 'axial_all_image', 'no_timeline', 'no_report', 'timeline_only', 'report', 'path', 'path_image_and_report', 'path_full', 'retrieved_timeline', ...
            storage_client: GCP Storage client for loading NIfTI files from bucket (used when file not under ct_dir).
            model_type: Model type string (e.g., 'gemma3') to determine preprocessing.
            ct_dir: Optional path from config paths.ct_dir. If set and the resolved CT filename ({study}__{series}.nii.gz) exists under ct_dir, load from disk; else use GCP.
            ct_snapshot_prefix: CT storage prefix (config paths.ct_snapshot_prefix). None -> DEFAULT_CT_SNAPSHOT_PREFIX (feb26). Threaded into the blob resolver so re-materialization is a config/data change.
        """
        self.df = df.reset_index(drop=True)
        self.prompt_col = prompt_col
        self.add_options = add_options
        self.experiment = experiment
        self.storage_client = storage_client
        self.ct_snapshot_prefix = ct_snapshot_prefix or DEFAULT_CT_SNAPSHOT_PREFIX
        self.bucket_name = 'su-vista-uscentral1'
        self.model_type = model_type
        self.is_gemma = model_type is not None and 'gemma' in model_type.lower()
        self.ct_dir = Path(ct_dir) if ct_dir else None

        # Determine image size based on model type
        self.target_size = 448 if self.is_gemma else 512
    
    def _process_ct_slice(self, ct_slice):
        """
        Process a CT slice: apply windowing for gemma models, normalize, convert to PIL Image, and resize.
        
        Args:
            ct_slice: numpy array of CT slice data
            
        Returns:
            PIL Image object
        """
        if self.is_gemma:
            # Apply windowing function for gemma models
            windowed_slice = multi_window_rgb(ct_slice)
            # Round slice voxels to nearest integer number
            windowed_slice = np.round(windowed_slice, 0).astype(np.uint8)
            # Convert to PIL Image (RGB mode since windowing creates 3 channels)
            pil_img = Image.fromarray(windowed_slice, mode='RGB')
        else:
            # Standard normalization for non-gemma models
            normalized_slice = grayscale(ct_slice)
            pil_img = Image.fromarray(normalized_slice, mode='L')
        
        # Resize to target size (448 for gemma, 512 for others)
        return pad_to_size(pil_img, self.target_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Handle Prompt
        question = str(row[self.prompt_col])
        
        if self.add_options and 'options' in row and pd.notna(row['options']):
            question = f"{question} Options: {row['options']}"

        # 2. Handle Image based on experiment type
        img = None
        # selected_indices records WHICH leaf units this item chose, so the golden-output
        # harness (src/vista_run/golden_harness.py) can diff selection byte-for-byte across
        # the Task-3b dissolution — CT: list of axial slice indices; pathology: list of
        # loaded tile paths; single-image / no-image: None. Additive + inference-neutral:
        # it does not touch `question` or `image`, only surfaces already-computed choices.
        selected_indices = None
        image_path = row.get('image_path', None)
        
        # Skip image loading for 'no_image', 'report', 'timeline_only', 'all_vb_timeline_only', 'retrieved_timeline', and 'retrieved_timeline_per_iteration' experiments
        # (retrieved_timeline_with_image and retrieved_timeline_per_iteration_summarization_with_image load 50 axial slices like axial_all_image)
        if self.experiment in ('no_image', 'report', 'timeline_only', 'all_vb_timeline_only', 'retrieved_timeline', 'retrieved_timeline_per_iteration', 'retrieved_timeline_per_iteration_summarization'):
            img = None
        else:
            # Path / path_image_and_report / path_full: load pathology tile paths from path_tile_paths (list)
            if self.experiment in ('path', 'path_image_and_report', 'path_full'):
                path_tile_paths = row.get('path_tile_paths', None)
                if path_tile_paths is not None and len(path_tile_paths) > 0:
                    img_list = []
                    loaded_tile_paths = []
                    for p in path_tile_paths:
                        p_str = str(p).strip()
                        if p_str and os.path.exists(p_str):
                            try:
                                with Image.open(p_str) as PIL_img:
                                    PIL_img.load()
                                    if PIL_img.mode != 'RGB':
                                        PIL_img = PIL_img.convert('RGB')
                                    img_list.append(pad_to_size(PIL_img.copy(), self.target_size))
                                    loaded_tile_paths.append(p_str)
                            except Exception as e:
                                pass
                    if img_list:
                        img = img_list
                        selected_indices = loaded_tile_paths

            # First check for image_path (existing behavior) if not path experiment or no path tiles
            if img is None and pd.notna(image_path) and os.path.exists(str(image_path)):
                try:
                    with Image.open(image_path) as PIL_img:
                        PIL_img.load()
                        img = pad_to_size(PIL_img.copy(), self.target_size)
                    print(f"[IMAGE] Using image from image_path: {image_path} (index: {row.get('index', idx)})")
                except Exception as e:
                    print(f"Error loading image at {image_path}: {e}")
            
            # If no image from image_path, resolve CT via the v1_5 (study, series) UID link.
            # `nifti_path` is a deprecated INTEGER in v1_5 — never read it; the go-forward
            # substrate links CTs by (image_study_uid, image_series_uid) -> feb26 blob.
            # Missing / null UIDs fail closed to no-image (resolve_ct_blob -> None).
            if img is None:
                study_uid = row.get('image_study_uid', None)
                series_uid = row.get('image_series_uid', None)
                resolved = resolve_ct_blob(study_uid, series_uid, prefix=self.ct_snapshot_prefix)
                if resolved is not None:
                    blob_path, filename = resolved
                    tmp_file_path = None
                    try:
                        import nibabel as nib
                        local_path = self.ct_dir / filename if self.ct_dir else None
                        use_local = local_path is not None and local_path.exists()
                        # Prefer reading the feb26 NIfTI directly off the gcsfuse mount
                        # (/mnt/<bucket>/<blob>) — no full download, and GCS Anywhere Cache
                        # accelerates repeat reads. Fall back to a storage-client download to a
                        # temp file only when the mount path is absent.
                        mount_path = Path("/mnt") / self.bucket_name / blob_path
                        use_mount = (not use_local) and mount_path.exists()
                        # Source-of-image log (rung-0 0a preflight seam): which route each CT row
                        # takes (local ct_dir / gcsfuse mount / GCS download) + the resolved blob,
                        # so the feb26 reroute stays verifiable and the (study, series) link
                        # traceable. All three routes read the same feb26 bytes. VM-side logs only.
                        if use_local:
                            _ct_source = "local"
                        elif use_mount:
                            _ct_source = "mount"
                        elif self.storage_client is not None:
                            _ct_source = "gcs"
                        else:
                            _ct_source = "none"
                        print(f"[CT] source={_ct_source} prefix={self.ct_snapshot_prefix} blob={blob_path} file={filename} local_exists={use_local} (index: {row.get('index', idx)})")

                        # Lazy image handle: slices are pulled via ct_img.dataobj[...] (below)
                        # instead of get_fdata(), so the full float64 volume is never materialized
                        # — a 512x512x8652 CT would need ~18 GB via get_fdata and OOM the box.
                        # (indexed_gzip gives efficient random access into the .nii.gz.)
                        if use_local:
                            ct_img = nib.load(str(local_path))
                        elif use_mount:
                            ct_img = nib.load(str(mount_path))
                        elif self.storage_client is not None:
                            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                                tmp_file_path = tmp_file.name
                            self.storage_client.bucket(self.bucket_name).blob(blob_path).download_to_filename(tmp_file_path)
                            ct_img = nib.load(tmp_file_path)
                        else:
                            ct_img = None

                        if ct_img is not None:
                            # Handle different experiment types
                            if self.experiment in ('axial_all_image', 'retrieved_timeline_with_image', 'retrieved_timeline_per_iteration_summarization_with_image'):
                                # 30 axial slices, evenly spaced across the volume depth.
                                # (Previously used a buggy i*0.1 scheme that overshot 1.0 for i>10 and
                                # silently clamped every later slice to the last; now matches the even
                                # spacing used by the 50- and 10-slice branches below.)
                                if len(ct_img.shape) > 2:
                                    depth = ct_img.shape[2]
                                    img_list = []
                                    slice_indices = []
                                    num_slices = 30
                                    for i in range(num_slices):
                                        if num_slices > 1:
                                            position = i / (num_slices - 1)
                                        else:
                                            position = 0.0
                                        index = int(position * (depth - 1))
                                        if index >= depth:
                                            index = depth - 1
                                        slice_indices.append(index)
                                        axial_slice = np.asarray(ct_img.dataobj[:, :, index], dtype=np.float64)
                                        img_list.append(self._process_ct_slice(axial_slice))
                                    img = img_list
                                    selected_indices = slice_indices
                                else:
                                    img = None
                            elif self.experiment in ('no_timeline', 'all_vb_image_only'):
                                # Both use 50 axial slices (image-only, no patient timeline)
                                if len(ct_img.shape) > 2:
                                    depth = ct_img.shape[2]
                                    img_list = []
                                    slice_indices = []
                                    num_slices = 50
                                    for i in range(num_slices):
                                        if num_slices > 1:
                                            position = i / (num_slices - 1)
                                        else:
                                            position = 0.0
                                        index = int(position * (depth - 1))
                                        if index >= depth:
                                            index = depth - 1
                                        slice_indices.append(index)
                                        axial_slice = np.asarray(ct_img.dataobj[:, :, index], dtype=np.float64)
                                        img_list.append(self._process_ct_slice(axial_slice))
                                    img = img_list
                                    selected_indices = slice_indices
                                else:
                                    img = None
                            elif self.experiment == 'no_report':
                                if len(ct_img.shape) > 2:
                                    depth = ct_img.shape[2]
                                    img_list = []
                                    slice_indices = []
                                    num_slices = 10
                                    for i in range(num_slices):
                                        if num_slices > 1:
                                            position = i / (num_slices - 1)
                                        else:
                                            position = 0.0
                                        index = int(position * (depth - 1))
                                        if index >= depth:
                                            index = depth - 1
                                        slice_indices.append(index)
                                        axial_slice = np.asarray(ct_img.dataobj[:, :, index], dtype=np.float64)
                                        img_list.append(self._process_ct_slice(axial_slice))
                                    img = img_list
                                    selected_indices = slice_indices
                                else:
                                    img = None
                    except Exception as e:
                        print(f"Error loading NIfTI for (study={study_uid}, series={series_uid}) blob={blob_path}: {e}")
                        # Continue with text-only if image loading fails
                    finally:
                        # Only the GCS-download route creates a temp file; the mount/local
                        # routes read in place. Unlink here (after lazy slicing) since dataobj
                        # reads from the file during slicing, not at nib.load() time.
                        if tmp_file_path is not None and os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)

        # 3. Build Item
        item = {
            "index": row.get("index", idx),
            "question": question,
            "image_path": image_path,
            "image": img,
            "options": row.get("options", None),
            # Leaf selections chosen above (golden-harness capture; None when no image).
            "selected_indices": selected_indices,
            # Pass the raw row so the Orchestrator can access all original metadata for saving
            "raw_row": row
        }

        return item

def prompt_collate(batch):
    """Returns the batch as a list of dictionaries."""
    return batch