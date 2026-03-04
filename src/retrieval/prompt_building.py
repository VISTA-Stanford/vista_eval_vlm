"""
Build dynamic_prompt for retrieval experiments: single timeline or per-iteration.
Used by vista_run/run_bq.py to keep prompt construction out of the main orchestrator.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from retrieval.iterative_retrieval import (
    run_iterative_retrieval_batch,
    _summarize_timeline_for_context_batch,
)
from data_tools.utils.meds_timeline_utils import truncate_timeline, count_timeline_events


# Experiment name sets for dispatch
RETRIEVAL_SINGLE_TIMELINE = frozenset({"retrieved_timeline"})
RETRIEVAL_PER_ITERATION = frozenset({
    "retrieved_timeline_per_iteration",
    "retrieved_timeline_per_iteration_summarization",
    "retrieved_timeline_with_image",
    "retrieved_timeline_per_iteration_summarization_with_image",
})


def _build_batch_data_from_rows(rows_list, retrieval_cfg):
    """Build list of dicts for run_iterative_retrieval_batch from (index, row) iterable."""
    batch_data = []
    use_time_filter = retrieval_cfg.get("use_time_filter", False)
    for _, r in rows_list:
        entry = {
            "person_id": str(r["person_id"]),
            "question": str(r.get("question", r.get("label_description", ""))),
        }
        if use_time_filter:
            embed_time = r.get("embed_time")
            if embed_time is not None and pd.notna(embed_time):
                et = pd.to_datetime(embed_time, errors="coerce")
                if pd.notna(et):
                    entry["end_date"] = et.strftime("%Y-%m-%d")
                    start_dt = et - pd.DateOffset(months=retrieval_cfg.get("months_before", 6))
                    entry["start_date"] = start_dt.strftime("%Y-%m-%d")
        batch_data.append(entry)
    return batch_data


def _write_retrieval_log_csv(log_rows, csv_log_path, task_name, file_model_name):
    """Write retrieval_keywords.csv from list of log row dicts."""
    if not log_rows:
        return
    df_log = pd.DataFrame(log_rows)
    df_log["task"] = task_name
    df_log["model"] = file_model_name
    df_log.to_csv(csv_log_path, mode="w", index=False)


def build_retrieval_prompts_single_timeline(
    df_exp,
    task_name,
    base_prompt_template,
    retriever,
    adapter,
    model,
    processor,
    retrieval_cfg,
    truncation_config,
    results_base,
    source_csv,
    file_model_name,
):
    """
    Build dynamic_prompt for each row using one retrieved timeline per patient.
    Returns df_exp with dynamic_prompt set.
    """
    max_rows = retrieval_cfg.get("max_rows")
    iterations_log_dir = retrieval_cfg.get("iterations_log_dir") or (
        results_base / "retrieval_logs" / source_csv / task_name / file_model_name
    )
    save_log = retrieval_cfg.get("save_iterations_log", True)

    if max_rows:
        df_exp = df_exp.head(max_rows).copy()
    if save_log:
        Path(iterations_log_dir).mkdir(parents=True, exist_ok=True)

    rows_list = list(df_exp.iterrows())
    batch_size = retrieval_cfg.get("retrieval_batch_size", 8)
    prompts_and_logs = []

    for i in tqdm(range(0, len(rows_list), batch_size), desc="Retrieval"):
        batch_rows = rows_list[i : i + batch_size]
        batch_data = _build_batch_data_from_rows(batch_rows, retrieval_cfg)
        results = run_iterative_retrieval_batch(
            retriever, adapter, model, processor,
            batch_data=batch_data,
            task_name=task_name,
            max_iterations=retrieval_cfg.get("max_iterations", 3),
            keywords_per_iteration=retrieval_cfg.get("keywords_per_iteration", 5),
            records_per_keyword=retrieval_cfg.get("records_per_keyword", 5),
            summarize_timeline_for_context=retrieval_cfg.get("summarize_timeline_for_context", False),
            timeline_summary_max_chars=retrieval_cfg.get("timeline_summary_max_chars", 4000),
        )
        for (_, row), result in zip(batch_rows, results):
            combined = truncate_timeline(result.timeline_str, truncation_config)
            prompts_and_logs.append((base_prompt_template.replace("[PATIENT_TIMELINE]", combined), result))

    df_exp["dynamic_prompt"] = [p for p, _ in prompts_and_logs]

    if save_log:
        csv_log_path = Path(iterations_log_dir) / "retrieval_keywords.csv"
        log_rows = []
        for (_, row), (_, result) in zip(df_exp.iterrows(), prompts_and_logs):
            for log_entry in result.iterations_log:
                log_rows.append({
                    "person_id": str(row["person_id"]),
                    "iteration": log_entry["iteration"],
                    "all_keywords_so_far": ", ".join(log_entry.get("all_keywords_so_far", [])),
                    "num_results": sum(log_entry.get("num_results_per_keyword", [])),
                    "total_unique": log_entry.get("total_unique_so_far", 0),
                    "internal_state": log_entry.get("internal_state", ""),
                    "current_evidence": log_entry.get("current_evidence", ""),
                    "search_history": log_entry.get("search_history", ""),
                    "answer": log_entry.get("answer", ""),
                    "clinical_reasoning": log_entry.get("clinical_reasoning", ""),
                    "raw_model_output": log_entry.get("raw_model_output", ""),
                    "summary_patient_timeline": log_entry.get("summary_patient_timeline", ""),
                })
        _write_retrieval_log_csv(log_rows, csv_log_path, task_name, file_model_name)

    return df_exp


def build_retrieval_prompts_per_iteration(
    df_exp,
    experiment,
    task_name,
    base_prompt_template,
    retriever,
    adapter,
    model,
    processor,
    retrieval_cfg,
    truncation_config,
    results_base,
    source_csv,
    file_model_name,
):
    """
    Build prompts and expand to one row per (patient, iteration).
    Returns dataframe with dynamic_prompt and index reset.
    """
    max_iterations = retrieval_cfg.get("max_iterations", 3)
    max_rows = retrieval_cfg.get("max_rows")
    iterations_log_dir = retrieval_cfg.get("iterations_log_dir") or (
        results_base / "retrieval_logs" / source_csv / task_name / file_model_name
    )
    save_log = retrieval_cfg.get("save_iterations_log", True)
    save_timeline_cache = retrieval_cfg.get("save_timeline_cache", False)
    use_timeline_cache = retrieval_cfg.get("use_timeline_cache", False)
    cache_path = Path(iterations_log_dir) / "retrieval_timelines_per_iteration.parquet"
    use_vlm_summary = experiment in (
        "retrieved_timeline_per_iteration_summarization",
        "retrieved_timeline_per_iteration_summarization_with_image",
    )

    if max_rows:
        df_exp = df_exp.head(max_rows).copy()
    if save_log or save_timeline_cache or use_timeline_cache:
        Path(iterations_log_dir).mkdir(parents=True, exist_ok=True)

    rows_list = list(df_exp.iterrows())

    if use_timeline_cache and cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
        summary_df = None
        if use_vlm_summary:
            csv_log_path = Path(iterations_log_dir) / "retrieval_keywords.csv"
            if csv_log_path.exists():
                try:
                    kw_df = pd.read_csv(csv_log_path)
                    if "summary_patient_timeline" in kw_df.columns and "person_id" in kw_df.columns and "iteration" in kw_df.columns:
                        summary_df = kw_df[["person_id", "iteration", "summary_patient_timeline"]].copy()
                        summary_df["person_id"] = summary_df["person_id"].astype(str)
                except Exception:
                    pass
        expanded_rows = []
        for idx, row in rows_list:
            row_dict = row.to_dict()
            orig_index = row_dict.get("index", idx)
            cache_rows = cache_df[
                (cache_df["person_id"].astype(str) == str(row["person_id"]))
                & (cache_df["index"] == orig_index)
            ].sort_values("iteration")
            for _, cr in cache_rows.iterrows():
                new_row = dict(row_dict)
                iter_num = int(cr["iteration"])
                timeline = cr["timeline"]
                if use_vlm_summary and summary_df is not None:
                    match = summary_df[
                        (summary_df["person_id"] == str(row["person_id"]))
                        & (summary_df["iteration"] == iter_num)
                    ]
                    if not match.empty:
                        summ = match.iloc[0]["summary_patient_timeline"]
                        timeline_for_prompt = str(summ).strip() if pd.notna(summ) and str(summ).strip() else timeline
                    else:
                        timeline_for_prompt = timeline
                else:
                    timeline_for_prompt = timeline
                combined = truncate_timeline(timeline_for_prompt, truncation_config)
                new_row["dynamic_prompt"] = base_prompt_template.replace("[PATIENT_TIMELINE]", combined)
                new_row["unique_events"] = int(cr.get("unique_events", count_timeline_events(timeline)))
                new_row["iteration"] = iter_num
                expanded_rows.append(new_row)
        df_exp = pd.DataFrame(expanded_rows)
    else:
        all_results = []
        batch_size = retrieval_cfg.get("retrieval_batch_size", 8)
        for i in tqdm(range(0, len(rows_list), batch_size), desc="Retrieval"):
            batch_rows = rows_list[i : i + batch_size]
            batch_data = _build_batch_data_from_rows(batch_rows, retrieval_cfg)
            results = run_iterative_retrieval_batch(
                retriever, adapter, model, processor,
                batch_data=batch_data,
                task_name=task_name,
                max_iterations=max_iterations,
                keywords_per_iteration=retrieval_cfg.get("keywords_per_iteration", 5),
                records_per_keyword=retrieval_cfg.get("records_per_keyword", 5),
                summarize_timeline_for_context=retrieval_cfg.get("summarize_timeline_for_context", False),
                timeline_summary_max_chars=retrieval_cfg.get("timeline_summary_max_chars", 4000),
                search_diary_all_iterations=use_vlm_summary,
            )
            for (_, row), result in zip(batch_rows, results):
                all_results.append((row, result))

        expanded_rows = []
        cache_rows = []
        summaries_by_patient_iter = {}
        if use_vlm_summary:
            timeline_summary_max_chars = retrieval_cfg.get("timeline_summary_max_chars", 20000)
            for iter_num in tqdm(range(1, max_iterations + 1), desc="VLM summarization"):
                timelines = []
                task_queries = []
                for row, result in all_results:
                    if iter_num <= len(result.timeline_per_iteration):
                        timelines.append(result.timeline_per_iteration[iter_num - 1])
                        task_queries.append(str(row.get("question", row.get("label_description", "")) or task_name))
                    else:
                        timelines.append("No evidence retrieved yet.")
                        task_queries.append(task_name)
                if timelines:
                    summaries = _summarize_timeline_for_context_batch(
                        adapter, model, processor,
                        timelines, task_queries,
                        max_chars=timeline_summary_max_chars,
                    )
                    for patient_idx, summ in enumerate(summaries):
                        summaries_by_patient_iter[(patient_idx, iter_num)] = summ

        for patient_idx, (row, result) in enumerate(all_results):
            row_dict = row.to_dict()
            orig_index = row_dict.get("index", row.name)
            timelines_full = result.timeline_per_iteration
            iterations_log = result.iterations_log
            for iter_num, timeline in enumerate(timelines_full, start=1):
                new_row = dict(row_dict)
                new_row["iteration"] = iter_num
                if use_vlm_summary and (patient_idx, iter_num) in summaries_by_patient_iter:
                    timeline_for_prompt = summaries_by_patient_iter[(patient_idx, iter_num)]
                else:
                    timeline_for_prompt = timeline
                combined = truncate_timeline(timeline_for_prompt, truncation_config)
                new_row["dynamic_prompt"] = base_prompt_template.replace("[PATIENT_TIMELINE]", combined)
                new_row["unique_events"] = count_timeline_events(timeline)
                expanded_rows.append(new_row)
                summary_pt = ""
                if iterations_log:
                    log_idx = min(iter_num, len(iterations_log) - 1)
                    summary_pt = iterations_log[log_idx].get("summary_patient_timeline", "") or ""
                cache_rows.append({
                    "person_id": str(row["person_id"]),
                    "index": orig_index,
                    "iteration": iter_num,
                    "timeline": timeline,
                    "timeline_summarized": None,
                    "summary_patient_timeline": summary_pt if summary_pt else None,
                    "unique_events": count_timeline_events(timeline),
                })

        df_exp = pd.DataFrame(expanded_rows)
        if save_timeline_cache and cache_rows:
            pd.DataFrame(cache_rows).to_parquet(cache_path, index=False)
        if save_log and all_results:
            csv_log_path = Path(iterations_log_dir) / "retrieval_keywords.csv"
            log_rows = []
            for row, result in all_results:
                for log_entry in result.iterations_log:
                    log_rows.append({
                        "person_id": str(row["person_id"]),
                        "iteration": log_entry["iteration"],
                        "all_keywords_so_far": ", ".join(log_entry.get("all_keywords_so_far", [])),
                        "num_results": sum(log_entry.get("num_results_per_keyword", [])),
                        "total_unique": log_entry.get("total_unique_so_far", 0),
                        "internal_state": log_entry.get("internal_state", ""),
                        "current_evidence": log_entry.get("current_evidence", ""),
                        "search_history": log_entry.get("search_history", ""),
                        "answer": log_entry.get("answer", ""),
                        "clinical_reasoning": log_entry.get("clinical_reasoning", ""),
                        "raw_model_output": log_entry.get("raw_model_output", ""),
                        "summary_patient_timeline": log_entry.get("summary_patient_timeline", ""),
                    })
            _write_retrieval_log_csv(log_rows, csv_log_path, task_name, file_model_name)

    df_exp["index"] = range(len(df_exp))
    return df_exp


def build_retrieval_prompts(
    experiment,
    df_exp,
    task_name,
    base_prompt_template,
    retriever,
    adapter,
    model,
    processor,
    retrieval_cfg,
    truncation_config,
    results_base,
    source_csv,
    file_model_name,
):
    """
    Build dynamic_prompt for retrieval experiments. Dispatches to single-timeline or per-iteration.
    Returns df_exp with dynamic_prompt set (and possibly expanded rows for per-iteration).
    """
    if experiment in RETRIEVAL_SINGLE_TIMELINE:
        return build_retrieval_prompts_single_timeline(
            df_exp, task_name, base_prompt_template,
            retriever, adapter, model, processor,
            retrieval_cfg, truncation_config,
            results_base, source_csv, file_model_name,
        )
    if experiment in RETRIEVAL_PER_ITERATION:
        return build_retrieval_prompts_per_iteration(
            df_exp, experiment, task_name, base_prompt_template,
            retriever, adapter, model, processor,
            retrieval_cfg, truncation_config,
            results_base, source_csv, file_model_name,
        )
    raise ValueError(f"Unknown retrieval experiment: {experiment}")
