import os
import argparse
import json
import yaml
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping
from queue import Queue
from threading import Thread
from tqdm import tqdm
from torch.utils.data import DataLoader
from google.cloud import bigquery
from google.cloud import storage
# import meds_reader
# from meds_tools import patient_timeline
# from meds2text.ontology import OntologyDescriptionLookupTable

from vqa_dataset import PromptDataset, prompt_collate
from models import load_model_adapter
from data_tools.utils.query_utils import VISTA_BENCH_DATASET, fetch_task_data_from_bq
from data_tools.utils.meds_timeline_utils import (
    count_unique_event_dates,
    truncate_timeline,
)
from data_tools.utils.task_data_utils import (
    resolve_local_bq_cache_path,
    resolve_timeline_csv_path,
    resolve_timeline_csv_filename,
    resolve_retrieval_csv_path,
    find_bq_timeline_column,
    merge_bq_with_timeline_csv,
)
from data_tools.utils.csv_utils import append_to_csv as append_to_csv_util
from retrieval.prompt_building import (
    build_retrieval_prompts,
    RETRIEVAL_SINGLE_TIMELINE,
    RETRIEVAL_PER_ITERATION,
)

# All retrieval experiments (used for overwrite-output and data-loading dispatch)
RETRIEVAL_EXPERIMENTS = RETRIEVAL_SINGLE_TIMELINE | RETRIEVAL_PER_ITERATION

# Columns to drop from raw_row when building result row (heavy or redundant)
HEAVY_RESULT_COLUMNS = ("note_text", "patient_string", "report")


class TaskOrchestrator:
    def __init__(self, config_path, model_type, model_name):
        # 1. Load YAML Config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        self.base_path = Path(self.cfg['paths']['base_dir'])
        self.results_base = Path(self.cfg['paths']['results_dir'])
        self.model_type = model_type
        self.model_name = model_name
        self.file_model_name = model_name.split('/')[-1].replace('/', '_')
        
        # 2. Set Environments
        self._set_envs(self.cfg['runtime']['cache_dir'])
        
        # 3. Load Task Registries
        valid_task_path = os.path.join(self.base_path, self.cfg['paths']['valid_tasks'])
        prompts_path = os.path.join(self.base_path, self.cfg['paths']['prompts'])

        with open(valid_task_path, 'r') as f:
            self.valid_tasks = json.load(f)
        with open(prompts_path, 'r') as f:
            self.prompts_map = json.load(f)
        prompts_summarized_path = os.path.join(
            self.base_path,
            self.cfg['paths'].get('prompts_summarized', 'tasks/prompts_by_task_summarized.json')
        )
        self.prompts_map_summarized = {}
        if os.path.exists(prompts_summarized_path):
            with open(prompts_summarized_path, 'r') as f:
                self.prompts_map_summarized = json.load(f)

        # Image-style prompts (used only for experiment 'no_timeline')
        image_prompts_path = os.path.join(
            self.base_path,
            self.cfg['paths'].get('image_prompts', 'tasks/image_valid_tasks.json')
        )
        self.image_prompts_map = {}
        if os.path.exists(image_prompts_path):
            with open(image_prompts_path, 'r') as f:
                self.image_prompts_map = json.load(f)

        # 4. Initialize BigQuery Client (NEW)
        # We default to the specific project mentioned. 
        self.project_id = "som-nero-plevriti-deidbdf"
        self.bq_client = bigquery.Client(project=self.project_id)
        
        # 4.25. Initialize GCP Storage Client for NIfTI images
        self.storage_client = storage.Client(project=self.project_id)

        # 5. Initialize Model
        self.adapter = load_model_adapter(
            self.model_type, 
            self.model_name, 
            self.cfg['model'].get("device", "auto"), 
            self.cfg['runtime']['cache_dir']
        )
        self.model, self.processor = self.adapter.load()

        # 6. Initialize retrieval (optional)
        retrieval_cfg = self.cfg.get("retrieval", {})
        if retrieval_cfg.get("enabled"):
            try:
                from retrieval import LocalPatientRetriever
                self.retriever = LocalPatientRetriever(
                    corpus_dir=retrieval_cfg["corpus_dir"],
                    cache_dir=retrieval_cfg["cache_dir"],
                )
            except ImportError as e:
                raise ImportError(
                    "retrieval.enabled is true but meds_mcp not installed. "
                    "Install with: pip install -e '.[retrieval]'"
                ) from e
        else:
            self.retriever = None
        self.retrieval_cfg = retrieval_cfg

    def _set_envs(self, model_dir):
        os.environ.update({
            "HF_HOME": model_dir,
            "TRANSFORMERS_CACHE": model_dir,
            "VLLM_CACHE_ROOT": model_dir,
            "TOKENIZERS_PARALLELISM": "false"
        })

    def _count_prompt_tokens(self, text: str) -> int:
        """Count token length of prompt text using model tokenizer."""
        if text is None:
            return 0
        text = str(text)
        tokenizer = getattr(self.processor, "tokenizer", None) if getattr(self, "processor", None) else None
        if tokenizer is None:
            tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except (TypeError, Exception):
                try:
                    return len(tokenizer.encode(text))
                except Exception:
                    pass
        return len(text.split())

    def run_inference(self, task_names=None):
        # Determine which tasks to run
        tasks_to_run = self.valid_tasks
        if task_names:
            tasks_to_run = [t for t in self.valid_tasks if t['task_name'] in task_names]

        # Get experiments from config (default to ['no_image'] if not specified)
        experiments = self.cfg.get('experiments', ['no_image'])

        # For each task: load data (different CSV for no_report/report), then run all experiments
        for task_info in tasks_to_run:
            task_name = task_info['task_name']
            # Data variants:
            # - no_report/timeline_only/report: use *_subsampled_no_img_report.csv AND require patient timeline merge
            # - no_timeline: use *_subsampled_no_img_report.csv BUT do NOT require patient timeline (image-only)
            # - retrieved_timeline: use *_subsampled_retrieval.csv directly (no BQ/timeline merge)
            # - all other experiments: use *_subsampled.csv and require patient timeline merge
            needs_report_timeline = any(e in ('no_report', 'timeline_only', 'report') for e in experiments)
            needs_no_timeline = 'no_timeline' in experiments
            needs_retrieved_timeline = any(e in RETRIEVAL_EXPERIMENTS for e in experiments)
            needs_all_vb_timeline = 'all_vb_timeline_only' in experiments
            needs_all_vb_image = 'all_vb_image_only' in experiments
            needs_normal = any(
                e not in ('no_report', 'timeline_only', 'report', 'no_timeline', 'all_vb_timeline_only', 'all_vb_image_only')
                and e not in RETRIEVAL_EXPERIMENTS
                for e in experiments
            )

            loaded_normal = self._load_task_data(task_info, use_no_report_csv=False, require_timeline=True) if needs_normal else None
            loaded_no_report = self._load_task_data(task_info, use_no_report_csv=True, require_timeline=True) if needs_report_timeline else None
            loaded_no_timeline = self._load_task_data(task_info, use_no_report_csv=True, require_timeline=False) if needs_no_timeline else None
            loaded_retrieval = self._load_retrieval_task_data(task_info) if needs_retrieved_timeline else None
            loaded_all_vb_timeline = self._load_all_vb_timeline_task_data(task_info) if needs_all_vb_timeline else None
            loaded_all_vb_image = self._load_all_vb_image_task_data(task_info) if needs_all_vb_image else None
            for experiment in experiments:
                print(f"\n>>> Starting Task: {task_name} | Experiment: {experiment}")
                if experiment == 'no_timeline':
                    loaded = loaded_no_timeline
                elif experiment == 'all_vb_image_only':
                    loaded = loaded_all_vb_image
                elif experiment == 'all_vb_timeline_only':
                    loaded = loaded_all_vb_timeline
                elif experiment in ('no_report', 'timeline_only', 'report'):
                    loaded = loaded_no_report
                elif experiment in RETRIEVAL_EXPERIMENTS:
                    loaded = loaded_retrieval
                else:
                    loaded = loaded_normal
                if loaded is None:
                    continue
                df, timeline_col, source_csv = loaded
                self._process_single_task_with_data(task_info, experiment, df, timeline_col)

    def _load_task_data(self, task_info, use_no_report_csv=False, require_timeline=True):
        """
        Query BigQuery once per task and merge with local CSV timelines.
        Returns (df, timeline_col, source_csv) or None on failure.
        Same data is reused for all experiments.
        When use_no_report_csv=True, loads from *_subsampled_no_img_report.csv (for no_report and report experiments).
        When require_timeline=False, we still use the local CSV to restrict to the subsampled cohort
        (via inner join on person_id), but we do NOT require a patient timeline column in the CSV.
        """
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']
        use_subsampled = self.cfg.get('subsample', False)

        # --- LOAD FROM LOCAL CACHE OR BIGQUERY ---
        local_bq_path = resolve_local_bq_cache_path(self.base_path, source_csv)
        if local_bq_path.exists():
            print(f"\n>>> Loading data for task: {task_name} (from local cache)")
            print(f"    Reading {local_bq_path}...")
            try:
                df_all = pd.read_csv(local_bq_path)
                df = df_all[df_all["task"] == task_name].copy()
            except Exception as e:
                print(f"!!! Error reading local BigQuery data for {task_name}: {e}")
                return None
            if df.empty:
                print(f"!!! No data found for task '{task_name}' in local file.")
                return None
            print(f"    Loaded {len(df)} rows from local cache.")
        else:
            full_table_id = f"{self.project_id}.{VISTA_BENCH_DATASET}.{source_csv}"
            print(f"\n>>> Loading data for task: {task_name} (one BigQuery query for all experiments)")
            print(f"    Querying BigQuery table: {full_table_id}...")
            df = fetch_task_data_from_bq(self.bq_client, full_table_id, task_name)
            if df is None or df.empty:
                print(f"!!! No data found for task '{task_name}' in BigQuery.")
                return None
            print(f"    Loaded {len(df)} rows from BigQuery.")

        if 'index' not in df.columns:
            df['index'] = df.index

        timeline_col = find_bq_timeline_column(df)
        if timeline_col not in df.columns:
            df[timeline_col] = None

        # --- LOAD PATIENT TIMELINES FROM LOCAL CSV ---
        csv_path = resolve_timeline_csv_path(
            self.base_path, source_csv, task_name, use_subsampled, use_no_report_csv
        )
        if not csv_path.exists():
            # Try v1_2 path: base_path/v1_2/source_csv/filename
            csv_path_v1_2 = self.base_path / "v1_2" / source_csv / resolve_timeline_csv_filename(
                task_name, use_subsampled, use_no_report_csv
            )
            if csv_path_v1_2.exists():
                csv_path = csv_path_v1_2
                print(f"    Using v1_2 path: {csv_path}")
            else:
                print(f"!!! Error: Local CSV file not found at {csv_path}")
                print(f"    Also checked v1_2 path: {csv_path_v1_2}")
                return None
        print(f"    Loading patient timelines from local CSV: {csv_path}")
        try:
            csv_df = pd.read_csv(csv_path)
            print(f"    Loaded {len(csv_df)} rows from local CSV.")
            if 'person_id' not in df.columns:
                print(f"!!! Error: 'person_id' column not found in BigQuery data.")
                return None
            if 'person_id' not in csv_df.columns:
                print(f"!!! Error: 'person_id' column not found in CSV data.")
                return None
            if require_timeline:
                df = merge_bq_with_timeline_csv(df, csv_df, timeline_col, use_no_report_csv)
                if df is None:
                    print(f"!!! Error: Patient timeline column not found in CSV or merge failed.")
                    return None
                matched_count = df[timeline_col].notna().sum()
                print(f"    Matched {matched_count} out of {len(df)} rows with patient timelines.")
            else:
                keep_ids = csv_df[['person_id']].drop_duplicates()
                df = df.merge(keep_ids, on='person_id', how='inner')
                print(f"    Restricted to {len(df)} rows via person_id join (no timeline merge).")
        except Exception as e:
            print(f"!!! Error loading/merging CSV: {e}")
            return None

        if require_timeline:
            df['unique_events'] = df[timeline_col].apply(count_unique_event_dates)
            truncation_config = self.cfg.get('timeline_truncation', None)
            df[timeline_col] = df[timeline_col].apply(lambda x: truncate_timeline(x, truncation_config))
        else:
            df['unique_events'] = float("nan")
        return (df, timeline_col, source_csv)

    def _load_retrieval_task_data(self, task_info):
        """
        Load task data directly from _subsampled_retrieval.csv (no BQ, no patient timeline merge).
        Used for retrieved_timeline experiment where timeline comes from retrieval.
        Returns (df, None, source_csv) or None on failure.
        """
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']

        csv_path = resolve_retrieval_csv_path(self.base_path, source_csv, task_name)
        if not csv_path.exists():
            # Fallback: try base_path without v1_2
            csv_path_alt = self.base_path / source_csv / f"{task_name}_subsampled_retrieval.csv"
            if csv_path_alt.exists():
                csv_path = csv_path_alt
            else:
                print(f"!!! Error: Retrieval CSV not found at {csv_path}")
                print(f"    Run format_retrieval_csv.py first to create _subsampled_retrieval.csv files.")
                return None

        print(f"\n>>> Loading data for task: {task_name} (retrieved_timeline – from _subsampled_retrieval.csv)")
        print(f"    Reading {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"!!! Error reading retrieval CSV for {task_name}: {e}")
            return None

        if df.empty:
            print(f"!!! No data found for task '{task_name}' in retrieval CSV.")
            return None

        print(f"    Loaded {len(df)} rows from retrieval CSV.")

        if 'index' not in df.columns:
            df['index'] = df.index

        df['unique_events'] = float("nan")
        return (df, None, source_csv)

    def _load_all_vb_timeline_task_data(self, task_info):
        """
        Load task data from {task_name}_full.parquet (full test split with patient_string).
        Used for all_vb_timeline_only experiment. Only requires patient_string.
        Returns (df, 'patient_string', source_csv) or None on failure.
        """
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']

        parquet_path = self.base_path / source_csv / f"{task_name}_full.parquet"
        if not parquet_path.exists():
            print(f"!!! Error: Full parquet not found at {parquet_path}")
            print(f"    Run full_test.py first to create _full.parquet files.")
            return None

        print(f"\n>>> Loading data for task: {task_name} (all_vb_timeline_only – from _full.parquet)")
        print(f"    Reading {parquet_path}...")
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"!!! Error reading parquet for {task_name}: {e}")
            return None

        if df.empty:
            print(f"!!! No data found for task '{task_name}' in parquet.")
            return None

        if 'patient_string' not in df.columns:
            print(f"!!! Error: 'patient_string' column not found in {parquet_path}")
            return None

        print(f"    Loaded {len(df)} rows from _full.parquet.")

        if 'index' not in df.columns:
            df['index'] = df.index

        timeline_col = 'patient_string'
        df['unique_events'] = df[timeline_col].apply(count_unique_event_dates)
        truncation_config = self.cfg.get('timeline_truncation', None)
        df[timeline_col] = df[timeline_col].apply(lambda x: truncate_timeline(x, truncation_config))
        return (df, timeline_col, source_csv)

    def _load_all_vb_image_task_data(self, task_info):
        """
        Load task data from {task_name}_full.parquet for all_vb_image_only experiment.
        Filters to rows with nifti_path (for image loading). Uses image prompts, no timeline.
        Returns (df, None, source_csv) - timeline_col is None.
        """
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']

        parquet_path = self.base_path / source_csv / f"{task_name}_full.parquet"
        if not parquet_path.exists():
            print(f"!!! Error: Full parquet not found at {parquet_path}")
            print(f"    Run full_test_ehr_vb_download.py first to create _full.parquet files.")
            return None

        print(f"\n>>> Loading data for task: {task_name} (all_vb_image_only – from _full.parquet)")
        print(f"    Reading {parquet_path}...")
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"!!! Error reading parquet for {task_name}: {e}")
            return None

        if df.empty:
            print(f"!!! No data found for task '{task_name}' in parquet.")
            return None

        if 'nifti_path' not in df.columns:
            print(f"!!! Error: 'nifti_path' column not found in {parquet_path}")
            return None
        if '_accession_number' not in df.columns:
            print(f"!!! Error: '_accession_number' column not found in {parquet_path}")
            return None

        # Filter to rows with nifti_path and _accession_number (required for image loading)
        df = df[
            df['nifti_path'].notna() & (df['nifti_path'].astype(str).str.strip() != "")
            & df['_accession_number'].notna() & (df['_accession_number'].astype(str).str.strip() != "")
        ].copy()
        if df.empty:
            print(f"!!! No rows with nifti_path and _accession_number for task '{task_name}'")
            return None

        print(f"    Loaded {len(df)} rows with nifti_path and _accession_number from _full.parquet.")

        if 'index' not in df.columns:
            df['index'] = df.index

        df['unique_events'] = float("nan")
        return (df, None, source_csv)

    def _prepare_batch_for_inference(self, batch, existing_indices, constrained_choices):
        """
        Prepare a batch for inference (CPU work: create_template, prepare_inputs).
        Returns (new_items, inference_batches, max_input_tokens) or None if no new items.
        inference_batches: list of {"indices": [...], "inputs": ...}
        """
        new_items = [item for item in batch if item['raw_row']['index'] not in existing_indices]
        if not new_items:
            return None

        tokenizer = getattr(self.processor, "tokenizer", None) if getattr(self, "processor", None) else None
        if tokenizer is None:
            tokenizer = getattr(self.model, "tokenizer", None)

        def _count_text_tokens(text: str) -> int:
            if text is None:
                return 0
            text = str(text)
            if tokenizer is not None and hasattr(tokenizer, "encode"):
                try:
                    return len(tokenizer.encode(text, add_special_tokens=False))
                except (TypeError, Exception):
                    try:
                        return len(tokenizer.encode(text))
                    except Exception:
                        pass
            return len(text.split())

        def _count_input_tokens(inp) -> int:
            if isinstance(inp, dict):
                ids = inp.get("input_ids", None)
                if ids is not None and hasattr(ids, "shape"):
                    try:
                        return int(ids.shape[-1]) if len(ids.shape) >= 2 else int(ids.shape[0])
                    except Exception:
                        pass
                if "prompt" in inp:
                    return _count_text_tokens(inp["prompt"])
                if "text" in inp:
                    return _count_text_tokens(inp["text"])
                return 0
            if isinstance(inp, tuple) and len(inp) >= 1:
                return _count_text_tokens(inp[0])
            if isinstance(inp, str):
                return _count_text_tokens(inp)
            return 0

        max_input_tokens = 0
        inference_batches = []

        if 'gemma' in self.model_type:
            def _get_image_group_key(item):
                image = item.get('image', None)
                if image is None:
                    return 0
                return len(image) if isinstance(image, list) else 1

            groups = defaultdict(list)
            for i, item in enumerate(new_items):
                groups[_get_image_group_key(item)].append((i, item))

            for key in sorted(groups.keys()):
                indexed_items = groups[key]
                indices = [x[0] for x in indexed_items]
                items = [x[1] for x in indexed_items]
                group_inputs = []
                for item in items:
                    try:
                        if not isinstance(item, dict):
                            raise TypeError(f"Expected item to be a dict, got {type(item)}")
                        single_msg = self.adapter.create_template(item)
                        single_inp = self.adapter.prepare_inputs([single_msg], self.processor, self.model)
                        if isinstance(single_inp, list):
                            single_inp = single_inp[0] if single_inp else {}
                        if isinstance(single_inp, Mapping) and not isinstance(single_inp, dict):
                            single_inp = dict(single_inp)
                        elif not isinstance(single_inp, dict):
                            raise TypeError(f"Expected dict or Mapping, got {type(single_inp)}")
                        group_inputs.append(single_inp)
                    except Exception as e:
                        print(f"Error preparing item: {e}")
                        group_inputs.append({})
                batched_inputs = self.adapter.stack_inputs(group_inputs, self.model)
                inference_batches.append({"indices": indices, "inputs": batched_inputs})
                for inp in group_inputs:
                    token_count = _count_input_tokens(inp)
                    if isinstance(inp, dict) and inp.get("multi_modal_data") and inp["multi_modal_data"].get("image"):
                        images = inp["multi_modal_data"]["image"]
                        num_images = len(images) if isinstance(images, list) else 1
                        token_count += num_images * 256
                    max_input_tokens = max(max_input_tokens, token_count)
        else:
            messages = [self.adapter.create_template(item) for item in new_items]
            inputs = self.adapter.prepare_inputs(messages, self.processor, self.model)
            inference_batches.append({"indices": list(range(len(new_items))), "inputs": inputs})
            if isinstance(inputs, list):
                for inp in inputs:
                    token_count = _count_input_tokens(inp)
                    # Add image tokens for vLLM-style inputs (Qwen3, InternVL, OctoMed, etc.)
                    if isinstance(inp, dict) and inp.get("multi_modal_data") and inp["multi_modal_data"].get("image"):
                        images = inp["multi_modal_data"]["image"]
                        num_images = len(images) if isinstance(images, list) else 1
                        token_count += num_images * 256  # Approximate tokens per image for VLMs
                    max_input_tokens = max(max_input_tokens, token_count)
            else:
                token_count = _count_input_tokens(inputs)
                if isinstance(inputs, dict) and inputs.get("multi_modal_data") and inputs["multi_modal_data"].get("image"):
                    images = inputs["multi_modal_data"]["image"]
                    num_images = len(images) if isinstance(images, list) else 1
                    token_count += num_images * 256
                max_input_tokens = max(max_input_tokens, token_count)

        return (new_items, inference_batches, max_input_tokens)

    def _setup_output_and_resume(self, task_info, experiment):
        """Create save dir, resolve out_file, optionally overwrite for retrieval, load existing indices for resume.
        Returns (out_file, existing_indices)."""
        source_csv = task_info['task_source_csv']
        task_name = task_info['task_name']
        save_dir = self.results_base / source_csv / task_name / self.file_model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{task_name}_results_{experiment}.csv"
        if experiment in RETRIEVAL_EXPERIMENTS:
            out_file.unlink(missing_ok=True)
        existing_indices = set()
        if out_file.exists():
            try:
                existing_df = pd.read_csv(out_file)
                if 'index' in existing_df.columns:
                    existing_indices = set(existing_df['index'].tolist())
                    print(f"Resuming: Found {len(existing_indices)} existing records.")
            except Exception as e:
                print(f"Could not read existing file, starting fresh: {e}")
        return out_file, existing_indices

    def _build_prompts_for_experiment(self, df, task_info, experiment, timeline_col):
        """Build dynamic_prompt column for the given experiment. Returns df_exp (copy or expanded)."""
        task_name = task_info['task_name']
        source_csv = task_info['task_source_csv']
        df_exp = df.copy()

        if experiment in ('no_timeline', 'all_vb_image_only'):
            base_prompt_template = self.image_prompts_map.get(task_name, "") or self.prompts_map.get(task_name, "")
            df_exp['dynamic_prompt'] = base_prompt_template
            return df_exp

        prompts_map = (
            self.prompts_map_summarized
            if experiment == 'retrieved_timeline_per_iteration_summarization'
            else self.prompts_map
        )
        base_prompt_template = prompts_map.get(task_name, "[PATIENT_TIMELINE]")

        if experiment == 'report':
            def timeline_with_report(row):
                timeline = str(row[timeline_col]) if pd.notna(row[timeline_col]) else ""
                report = str(row['report']) if 'report' in row and pd.notna(row.get('report')) else ""
                combined = f"{timeline}\nRadiology Report: {report}".rstrip()
                return base_prompt_template.replace('[PATIENT_TIMELINE]', combined)
            df_exp['dynamic_prompt'] = df_exp.apply(timeline_with_report, axis=1)
            return df_exp

        if experiment in RETRIEVAL_EXPERIMENTS:
            if self.retriever is None:
                raise ValueError(
                    f"{experiment} experiment requires retrieval.enabled=true in config"
                )
            truncation_config = None if experiment in RETRIEVAL_PER_ITERATION else self.cfg.get("timeline_truncation", None)
            return build_retrieval_prompts(
                experiment, df_exp, task_name, base_prompt_template,
                self.retriever, self.adapter, self.model, self.processor,
                self.retrieval_cfg, truncation_config,
                self.results_base, source_csv, self.file_model_name,
            )

        df_exp['dynamic_prompt'] = df_exp[timeline_col].apply(
            lambda x: base_prompt_template.replace('[PATIENT_TIMELINE]', str(x))
        )
        return df_exp

    def _build_result_row(self, item, out, experiment, timeline_col):
        """Build a single result row dict for CSV (drop heavy cols, add model_response, used_image, etc.)."""
        raw = item['raw_row']
        drop_cols = [c for c in [timeline_col] + list(HEAVY_RESULT_COLUMNS) if c and c in raw.index]
        res_row = raw.drop(labels=drop_cols, errors='ignore').to_dict()

        if isinstance(out, dict):
            res_row['model_response'] = out.get("text", "")
            res_row['cumulative_logprob'] = out.get("cumulative_logprob")
            res_row['log_probs'] = out.get("log_probs")
        else:
            res_row['model_response'] = out
            res_row['cumulative_logprob'] = None
            res_row['log_probs'] = None

        image = item.get('image', None)
        res_row['used_image'] = (1 if (image and (len(image) if isinstance(image, list) else 1)) else 0)

        if experiment in RETRIEVAL_PER_ITERATION:
            res_row['input_token_length'] = self._count_prompt_tokens(item.get('question', ''))
            if experiment in ('retrieved_timeline_with_image', 'retrieved_timeline_per_iteration_summarization_with_image') and image is not None:
                num_images = len(image) if isinstance(image, list) else 1
                res_row['input_token_length'] = res_row['input_token_length'] + num_images * 256
        if experiment == 'no_image' and timeline_col and timeline_col in raw:
            timeline_text = raw[timeline_col]
            res_row['timeline_token_count'] = self._count_prompt_tokens(
                str(timeline_text) if pd.notna(timeline_text) and str(timeline_text) != 'nan' else ''
            )
        return res_row

    def _run_inference_loop(
        self, loader, existing_indices, constrained_choices, out_file,
        task_name, experiment, timeline_col,
    ):
        """Producer-consumer inference loop: prepare batches in thread, run inference, build result rows, flush to CSV."""
        results_buffer = []
        batch_counter = 0
        prefetch_queue = Queue(maxsize=2)
        sentinel = object()

        def producer():
            try:
                for batch in loader:
                    prepared = self._prepare_batch_for_inference(batch, existing_indices, constrained_choices)
                    if prepared is not None:
                        prefetch_queue.put(prepared)
            except Exception as e:
                print(f"Producer error: {e}")
            finally:
                prefetch_queue.put(sentinel)

        producer_thread = Thread(target=producer, daemon=True)
        producer_thread.start()
        pbar = tqdm(desc=f"Inference {task_name} | {experiment}")

        while True:
            got = prefetch_queue.get()
            if got is sentinel:
                break
            new_items, inference_batches, max_input_tokens = got
            try:
                outputs = self._run_inference_on_batches(inference_batches, constrained_choices)
                for item, out in zip(new_items, outputs):
                    results_buffer.append(self._build_result_row(item, out, experiment, timeline_col))
                batch_counter += 1
                pbar.update(1)
                print(f"Batch {batch_counter}: Max input tokens = {max_input_tokens}")
                if batch_counter % 10 == 0:
                    append_to_csv_util(out_file, results_buffer)
                    results_buffer = []
                if batch_counter % 20 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in batch: {e}")
                print(f"Max input tokens = {max_input_tokens} (error occurred)")

        pbar.close()
        producer_thread.join(timeout=5)
        if results_buffer:
            append_to_csv_util(out_file, results_buffer)

    def _run_inference_on_batches(self, inference_batches, constrained_choices):
        """Run inference on prepared batches, return outputs in order of indices."""
        if not inference_batches:
            return []
        max_idx = max(i for b in inference_batches for i in b["indices"])
        output_list = [""] * (max_idx + 1)
        for batch in inference_batches:
            indices = batch["indices"]
            inputs = batch["inputs"]
            outputs = self.adapter.infer(
                self.model, self.processor, inputs,
                self.cfg['runtime']["max_new_tokens"],
                constrained_choices=constrained_choices
            )
            for i, out in zip(indices, outputs):
                output_list[i] = out
        return output_list

    def _process_single_task_with_data(self, task_info, experiment, df, timeline_col):
        """Run inference for one (task, experiment) using already-loaded task data."""
        task_name = task_info['task_name']
        constrained_choices = ["Yes", "No"] if task_info.get("is_binary", False) else None

        out_file, existing_indices = self._setup_output_and_resume(task_info, experiment)
        df_exp = self._build_prompts_for_experiment(df, task_info, experiment, timeline_col)

        ct_dir = self.cfg.get('paths', {}).get('ct_dir')
        dataset = PromptDataset(
            df=df_exp, prompt_col='dynamic_prompt', experiment=experiment,
            storage_client=self.storage_client, model_type=self.model_type, ct_dir=ct_dir,
        )
        num_workers = 4
        loader = DataLoader(
            dataset,
            batch_size=self.cfg['runtime']['batch_size'],
            shuffle=False,
            prefetch_factor=8,
            collate_fn=prompt_collate,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

        self._run_inference_loop(
            loader, existing_indices, constrained_choices, out_file,
            task_name, experiment, timeline_col,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--tasks", nargs='*', default=None)
    args = parser.parse_args()

    orchestrator = TaskOrchestrator(args.config, args.type, args.name)
    # Use command line tasks if provided, otherwise use config tasks
    tasks_to_run = args.tasks if args.tasks else orchestrator.cfg.get('tasks', None)
    orchestrator.run_inference(tasks_to_run)