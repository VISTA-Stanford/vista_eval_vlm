"""
Iterative VLM-driven patient timeline retrieval.

Fixed iterations, keywords_per_iteration per step (from config), records_per_keyword per keyword, no VLM decision.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from retrieval.format_events import format_retrieved_events
from retrieval.keyword_prompts import KEYWORD_EXTRACTION_TEMPLATE, TIMELINE_SUMMARY_TEMPLATE

if TYPE_CHECKING:
    from retrieval.local_retriever import LocalPatientRetriever

logger = logging.getLogger(__name__)


def _run_vlm_text_only(
    adapter: Any,
    model: Any,
    processor: Any,
    prompt: str,
    max_tokens: int = 1024,
) -> str:
    """
    Run VLM inference with text-only prompt (no image).
    """
    item = {"question": prompt}
    messages = [adapter.create_template(item)]
    inputs = adapter.prepare_inputs(messages, processor, model)
    outputs = adapter.infer(model, processor, inputs, max_tokens)
    if isinstance(outputs, list) and outputs:
        return str(outputs[0]).strip()
    return ""


def _run_vlm_batch(
    adapter: Any,
    model: Any,
    processor: Any,
    prompts: List[str],
    max_tokens: int = 1024,
) -> List[str]:
    """
    Run VLM inference with a batch of text-only prompts.
    Returns list of raw output strings (one per prompt).
    """
    if not prompts:
        return []
    messages = [adapter.create_template({"question": p}) for p in prompts]
    inputs = adapter.prepare_inputs(messages, processor, model)
    outputs = adapter.infer(model, processor, inputs, max_tokens)
    if not isinstance(outputs, list):
        return [""] * len(prompts)
    return [str(o).strip() if o is not None else "" for o in outputs]


def _summarize_timeline_for_context(
    adapter: Any,
    model: Any,
    processor: Any,
    timeline_str: str,
    task_query: str,
    max_chars: int = 4000,
    summary_template: Optional[str] = None,
) -> str:
    """
    Summarize a patient timeline for use as current_evidence in the next retrieval iteration.
    Returns summarized string, or original timeline on failure.
    """
    if not timeline_str or not timeline_str.strip():
        return timeline_str
    tpl = summary_template or TIMELINE_SUMMARY_TEMPLATE
    prompt = tpl.format(
        task_query=task_query,
        patient_timeline=timeline_str,
        max_chars=max_chars,
    )
    try:
        out = _run_vlm_text_only(adapter, model, processor, prompt, max_tokens=1024)
        if out and out.strip():
            extracted = _extract_answer(out)
            if extracted and extracted.strip():
                return extracted.strip()[:max_chars * 2]  # allow some overflow
            return out.strip()[:max_chars * 2]  # fallback if no <answer> tag
    except Exception as e:
        logger.warning("Timeline summarization failed: %s, using raw timeline", e)
    return timeline_str


def _summarize_timeline_for_context_batch(
    adapter: Any,
    model: Any,
    processor: Any,
    timelines: List[str],
    task_queries: List[str],
    max_chars: int = 4000,
    summary_template: Optional[str] = None,
) -> List[str]:
    """
    Batch summarize patient timelines for use as current_evidence.
    Returns list of summarized strings (or original on failure), same length as input.
    Skips summarization for empty or "No evidence retrieved yet." timelines.
    """
    if not timelines:
        return []
    EMPTY_MARKER = "No evidence retrieved yet."
    to_summarize = [
        (i, t, q)
        for i, (t, q) in enumerate(zip(timelines, task_queries))
        if t and t.strip() and t.strip() != EMPTY_MARKER
    ]
    if not to_summarize:
        return list(timelines)
    tpl = summary_template or TIMELINE_SUMMARY_TEMPLATE
    indices, t_vals, q_vals = zip(*to_summarize)
    prompts = [
        tpl.format(task_query=q, patient_timeline=t, max_chars=max_chars)
        for t, q in zip(t_vals, q_vals)
    ]
    result = list(timelines)
    try:
        outputs = _run_vlm_batch(adapter, model, processor, prompts, max_tokens=1024)
        for idx, out in zip(indices, outputs):
            if out and out.strip():
                extracted = _extract_answer(out)
                if extracted and extracted.strip():
                    result[idx] = extracted.strip()[:max_chars * 2]
                else:
                    result[idx] = out.strip()[:max_chars * 2]  # fallback if no <answer> tag
            # else keep original
    except Exception as e:
        logger.warning("Batch timeline summarization failed: %s, using raw timelines", e)
    return result


def _extract_tag(text: str, tag: str) -> str:
    """Extract content from <tag>...</tag>. Returns empty string if not found."""
    if not text:
        return ""
    match = re.search(
        rf"<{tag}\s*>(.+?)</{tag}\s*>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_answer(text: str) -> str:
    """
    Extract content from <answer>...</answer>. If closing tag is missing,
    extract everything after <answer> to end of text.
    """
    if not text:
        return ""
    # Normal case: both tags present
    content = _extract_tag(text, "answer")
    if content:
        return content
    # Fallback: <answer> present but </answer> missing - extract rest after <answer>
    match = re.search(
        r"<answer\s*>(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _parse_structured_output(raw: str, max_keywords: int = 5) -> Dict[str, Any]:
    """
    Parse model output for tags: internal_state, current_evidence, search_history, answer.
    Returns dict with keys: internal_state, current_evidence, search_history, keywords, reasoning.
    Falls back to <clinical_reasoning> for reasoning if present.
    """
    result: Dict[str, Any] = {
        "internal_state": "",
        "current_evidence": "",
        "search_history": "",
        "answer": "",
        "clinical_reasoning": "",
        "keywords": [],
        "reasoning": "",
    }
    if not raw or not str(raw).strip():
        return result

    text = str(raw).strip()

    # Extract structured tags
    result["internal_state"] = _extract_tag(text, "internal_state")
    result["current_evidence"] = _extract_tag(text, "current_evidence")
    result["search_history"] = _extract_tag(text, "search_history")
    answer_content = _extract_answer(text)
    result["answer"] = answer_content
    result["clinical_reasoning"] = _extract_tag(text, "clinical_reasoning")

    # Fallback: clinical_reasoning for reasoning
    result["reasoning"] = result["clinical_reasoning"]
    if not result["reasoning"] and result["internal_state"]:
        result["reasoning"] = result["internal_state"]

    # Parse keywords from <answer>
    keywords: List[str] = []
    if answer_content:
        try:
            keywords = json.loads(answer_content)
            if isinstance(keywords, list):
                keywords = [str(k).strip() for k in keywords if k][:max_keywords]
        except json.JSONDecodeError:
            keywords = [
                k.strip().strip('"\'') for k in re.findall(r'["\']([^"\']*)["\']', answer_content)
            ][:max_keywords]
            if not keywords:
                keywords = [p.strip().strip('"\'') for p in answer_content.split(",") if p.strip()][:max_keywords]

    if not keywords:
        parts = [p.strip().strip('"\'') for p in text.split(",") if p.strip()]
        keywords = parts[:max_keywords]

    keywords = [k for k in keywords if "<" not in k and ">" not in k and len(k) <= 50]
    result["keywords"] = keywords[:max_keywords]

    return result


def _ensure_n_keywords(
    keywords: List[str],
    task_name: str,
    question: str,
    n: int,
    exclude_searched: Optional[set] = None,
) -> List[str]:
    """Ensure exactly n keywords; pad with fallbacks if fewer, truncate if more.
    exclude_searched: optional set of lowercased keywords already searched (to avoid repeats).
    """
    exclude = exclude_searched or set()
    keywords = [k for k in keywords if k and k.lower() not in exclude]
    if len(keywords) >= n:
        return keywords[:n]
    fallbacks = [task_name] if task_name else []
    if question:
        words = [w for w in question.split() if len(w) > 3][:n]
        fallbacks.extend(words)
    while len(keywords) < n and fallbacks:
        candidate = fallbacks.pop(0)
        if candidate and candidate not in keywords and candidate.lower() not in exclude:
            keywords.append(candidate)
    return keywords[:n]


@dataclass
class IterativeRetrievalResult:
    """Result of iterative retrieval."""

    timeline_str: str
    timeline_per_iteration: List[str] = field(default_factory=list)  # timeline after each iteration
    timeline_per_iteration_summarized: List[str] = field(default_factory=list)  # short version (no VALUE) per iteration
    iterations_log: List[Dict[str, Any]] = field(default_factory=list)
    all_keywords: List[str] = field(default_factory=list)
    keyword_reasoning: List[str] = field(default_factory=list)


def run_iterative_retrieval(
    retriever: "LocalPatientRetriever",
    vlm_adapter: Any,
    vlm_model: Any,
    vlm_processor: Any,
    person_id: str,
    task_name: str,
    question: str,
    max_iterations: int = 3,
    keywords_per_iteration: int = 5,
    records_per_keyword: int = 5,
    keyword_extraction_template: Optional[str] = None,
    summarize_timeline_for_context: bool = False,
    timeline_summary_max_chars: int = 4000,
) -> IterativeRetrievalResult:
    """
    Run iterative retrieval: fixed iterations, keywords_per_iteration keywords, records_per_keyword per keyword.

    No VLM decision; always runs full max_iterations.
    Keywords regenerated each iteration using retrieved timeline as context.
    """
    kw_tpl = keyword_extraction_template or KEYWORD_EXTRACTION_TEMPLATE

    person_id_str = str(person_id).strip()
    all_results: List[Dict[str, Any]] = []
    seen_ids: set = set()
    iterations_log: List[Dict[str, Any]] = []
    all_keywords_flat: List[str] = []
    keyword_reasoning_list: List[str] = []

    prev_timeline = ""
    prev_internal_state = ""
    searched_keywords_list: List[str] = []
    timeline_per_iteration: List[str] = []
    timeline_per_iteration_summarized: List[str] = []
    clinical_reasoning_history: List[str] = []

    for iteration in range(1, max_iterations + 1):
        # 1. Build prompt with previous iteration's context (internal_state, patient_timeline, searched_keywords)
        task_query = question or task_name
        raw_timeline = prev_timeline if prev_timeline else "No evidence retrieved yet."
        if summarize_timeline_for_context and prev_timeline and prev_timeline.strip():
            previous_patient_timeline = _summarize_timeline_for_context(
                vlm_adapter, vlm_model, vlm_processor,
                raw_timeline, task_query, max_chars=timeline_summary_max_chars,
            )
        else:
            previous_patient_timeline = raw_timeline
        previous_searched_keywords = (
            ", ".join(dict.fromkeys(searched_keywords_list)) if searched_keywords_list else "No previous searches."
        )
        # search_diary: up to last 5 iterations of clinical_reasoning (empty on first iteration)
        last_5_reasoning = clinical_reasoning_history[-5:] if clinical_reasoning_history else []
        search_diary_parts = [
            f"--- Retrieval Iteration {j} Reasoning ---\n{cr}" for j, cr in enumerate(last_5_reasoning, start=iteration - len(last_5_reasoning))
        ]
        search_diary = "\n\n".join(search_diary_parts) if search_diary_parts else "(none yet)"

        format_kwargs: Dict[str, Any] = {
            "task_query": task_query,
            "patient_timeline": previous_patient_timeline,
            "searched_keywords": previous_searched_keywords,
            "iteration": iteration,
        }
        if "{search_diary}" in kw_tpl:
            format_kwargs["search_diary"] = search_diary
        if "{internal_state}" in kw_tpl:
            format_kwargs["internal_state"] = prev_internal_state or "None."
        prompt = kw_tpl.format(**format_kwargs)

        reasoning = ""
        try:
            kw_raw = _run_vlm_text_only(
                vlm_adapter, vlm_model, vlm_processor, prompt, max_tokens=1024
            )
        except Exception as e:
            logger.warning("Keyword extraction failed: %s, using fallback", e)
            kw_raw = ""
            reasoning = f"Fallback: {e}"

        parsed = _parse_structured_output(kw_raw, keywords_per_iteration)
        keywords = parsed["keywords"]
        if parsed["reasoning"]:
            reasoning = parsed["reasoning"]
        prev_internal_state = parsed["internal_state"]
        if parsed["clinical_reasoning"]:
            clinical_reasoning_history.append(parsed["clinical_reasoning"])

        exclude_searched = {k.lower() for k in searched_keywords_list}
        keywords = list(dict.fromkeys(keywords))
        keywords = _ensure_n_keywords(
            keywords, task_name, question, keywords_per_iteration, exclude_searched=exclude_searched
        )

        keyword_reasoning_list.append(reasoning)
        all_keywords_flat.extend(keywords)
        searched_keywords_list = list(dict.fromkeys(searched_keywords_list + keywords))

        # 2. Search per keyword (top 5 records each)
        results_this_iter: List[Dict[str, Any]] = []
        num_per_keyword: List[int] = []

        for kw in keywords:
            res = retriever.search(
                person_id=person_id_str,
                query=kw,
                max_results=records_per_keyword,
            )
            num_per_keyword.append(len(res))
            for r in res:
                rid = r.get("id")
                if rid is not None:
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_results.append(r)
                        results_this_iter.append(r)
                else:
                    all_results.append(r)
                    results_this_iter.append(r)

        total_unique = len(all_results)

        # 3. Format timeline for next iteration context
        # Short version (no VALUE content) for model prompt; full version for logging
        full_timeline = format_retrieved_events(all_results, exclude_report=False)
        short_timeline = format_retrieved_events(all_results, exclude_report=False, exclude_value=True)
        prev_timeline = short_timeline
        timeline_per_iteration.append(full_timeline)
        timeline_per_iteration_summarized.append(full_timeline)

        iterations_log.append({
            "iteration": iteration,
            "keywords": keywords,
            "all_keywords_so_far": list(dict.fromkeys(searched_keywords_list)),
            "num_results_per_keyword": num_per_keyword,
            "total_unique_so_far": total_unique,
            "keyword_reasoning": reasoning,
            "raw_model_output": kw_raw,
            "internal_state": parsed["internal_state"],
            "current_evidence": parsed["current_evidence"],
            "search_history": parsed["search_history"],
            "answer": parsed["answer"],
            "clinical_reasoning": parsed["clinical_reasoning"],
            "summary_patient_timeline": previous_patient_timeline if summarize_timeline_for_context else "",
        })

    timeline_str = format_retrieved_events(all_results, exclude_report=False)
    return IterativeRetrievalResult(
        timeline_str=timeline_str,
        timeline_per_iteration=timeline_per_iteration,
        timeline_per_iteration_summarized=timeline_per_iteration_summarized,
        iterations_log=iterations_log,
        all_keywords=list(dict.fromkeys(all_keywords_flat)),
        keyword_reasoning=keyword_reasoning_list,
    )


def run_iterative_retrieval_batch(
    retriever: "LocalPatientRetriever",
    vlm_adapter: Any,
    vlm_model: Any,
    vlm_processor: Any,
    batch_data: List[Dict[str, str]],
    task_name: str,
    max_iterations: int = 3,
    keywords_per_iteration: int = 5,
    records_per_keyword: int = 5,
    keyword_extraction_template: Optional[str] = None,
    summarize_timeline_for_context: bool = False,
    timeline_summary_max_chars: int = 4000,
) -> List[IterativeRetrievalResult]:
    """
    Run iterative retrieval for a batch of patients. Batches VLM keyword extraction
    across patients per iteration for efficiency.

    Args:
        batch_data: List of dicts with keys person_id and question.
        task_name: Task name for fallback keywords.

    Returns:
        List of IterativeRetrievalResult (one per patient, same order as batch_data).
    """
    kw_tpl = keyword_extraction_template or KEYWORD_EXTRACTION_TEMPLATE
    n = len(batch_data)

    # Per-patient state
    all_results_per_patient: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    seen_ids_per_patient: List[set] = [set() for _ in range(n)]
    iterations_log_per_patient: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    all_keywords_flat_per_patient: List[List[str]] = [[] for _ in range(n)]
    keyword_reasoning_per_patient: List[List[str]] = [[] for _ in range(n)]
    prev_timeline_per_patient: List[str] = [""] * n
    prev_internal_state_per_patient: List[str] = [""] * n
    searched_keywords_per_patient: List[List[str]] = [[] for _ in range(n)]
    timeline_per_iteration_per_patient: List[List[str]] = [[] for _ in range(n)]
    timeline_per_iteration_summarized_per_patient: List[List[str]] = [[] for _ in range(n)]
    clinical_reasoning_history_per_patient: List[List[str]] = [[] for _ in range(n)]

    for iteration in range(1, max_iterations + 1):
        # 1. Build prompts for all patients (internal_state, patient_timeline, searched_keywords, search_diary)
        # Compute previous_patient_timeline per patient (optionally summarized)
        raw_timelines = [
            prev_timeline_per_patient[i] if prev_timeline_per_patient[i] else "No evidence retrieved yet."
            for i in range(n)
        ]
        task_queries = [
            str(batch_data[i].get("question", "")).strip() or task_name
            for i in range(n)
        ]
        if summarize_timeline_for_context and any(t and t.strip() and t.strip() != "No evidence retrieved yet." for t in raw_timelines):
            previous_timelines = _summarize_timeline_for_context_batch(
                vlm_adapter, vlm_model, vlm_processor,
                raw_timelines, task_queries,
                max_chars=timeline_summary_max_chars,
            )
        else:
            previous_timelines = raw_timelines

        prompts: List[str] = []
        for i in range(n):
            person_id = str(batch_data[i].get("person_id", "")).strip()
            question = str(batch_data[i].get("question", "")).strip()
            task_query = question or task_name
            previous_patient_timeline = previous_timelines[i]
            previous_searched_keywords = (
                ", ".join(dict.fromkeys(searched_keywords_per_patient[i]))
                if searched_keywords_per_patient[i]
                else "No previous searches."
            )
            # search_diary: up to last 5 iterations of clinical_reasoning (empty on first iteration)
            last_5_reasoning = clinical_reasoning_history_per_patient[i][-5:] if clinical_reasoning_history_per_patient[i] else []
            search_diary_parts = [
                f"--- Retrieval Iteration {j} Reasoning ---\n{cr}" for j, cr in enumerate(last_5_reasoning, start=iteration - len(last_5_reasoning))
            ]
            search_diary = "\n\n".join(search_diary_parts) if search_diary_parts else "(none yet)"

            format_kwargs: Dict[str, Any] = {
                "task_query": task_query,
                "patient_timeline": previous_patient_timeline,
                "searched_keywords": previous_searched_keywords,
                "iteration": iteration,
            }
            if "{search_diary}" in kw_tpl:
                format_kwargs["search_diary"] = search_diary
            if "{internal_state}" in kw_tpl:
                format_kwargs["internal_state"] = prev_internal_state_per_patient[i] or "None."
            prompt = kw_tpl.format(**format_kwargs)
            prompts.append(prompt)

        # 2. Single VLM batch call
        try:
            kw_raw_list = _run_vlm_batch(
                vlm_adapter, vlm_model, vlm_processor,
                prompts, max_tokens=1024
            )
        except Exception as e:
            logger.warning("Batch keyword extraction failed: %s, using fallback", e)
            kw_raw_list = [""] * n

        # 3. Parse keywords per patient and run BM25
        for i in range(n):
            person_id_str = str(batch_data[i].get("person_id", "")).strip()
            question = str(batch_data[i].get("question", "")).strip()
            kw_raw = kw_raw_list[i] if i < len(kw_raw_list) else ""

            parsed = _parse_structured_output(kw_raw, keywords_per_iteration)
            keywords = parsed["keywords"]
            reasoning = parsed["reasoning"]
            prev_internal_state_per_patient[i] = parsed["internal_state"]
            if parsed["clinical_reasoning"]:
                clinical_reasoning_history_per_patient[i].append(parsed["clinical_reasoning"])

            exclude_searched = {k.lower() for k in searched_keywords_per_patient[i]}
            keywords = list(dict.fromkeys(keywords))
            keywords = _ensure_n_keywords(
                keywords, task_name, question, keywords_per_iteration, exclude_searched=exclude_searched
            )

            keyword_reasoning_per_patient[i].append(reasoning)
            all_keywords_flat_per_patient[i].extend(keywords)
            searched_keywords_per_patient[i] = list(
                dict.fromkeys(searched_keywords_per_patient[i] + keywords)
            )

            # BM25 search per keyword
            num_per_keyword: List[int] = []
            start_date = batch_data[i].get("start_date")
            end_date = batch_data[i].get("end_date")
            for kw in keywords:
                res = retriever.search(
                    person_id=person_id_str,
                    query=kw,
                    max_results=records_per_keyword,
                    start_date=start_date,
                    end_date=end_date,
                )
                num_per_keyword.append(len(res))
                for r in res:
                    rid = r.get("id")
                    if rid is not None:
                        if rid not in seen_ids_per_patient[i]:
                            seen_ids_per_patient[i].add(rid)
                            all_results_per_patient[i].append(r)
                    else:
                        all_results_per_patient[i].append(r)

            total_unique = len(all_results_per_patient[i])
            iterations_log_per_patient[i].append({
                "iteration": iteration,
                "keywords": keywords,
                "all_keywords_so_far": list(dict.fromkeys(searched_keywords_per_patient[i])),
                "num_results_per_keyword": num_per_keyword,
                "total_unique_so_far": total_unique,
                "raw_model_output": kw_raw,
                "internal_state": parsed["internal_state"],
                "current_evidence": parsed["current_evidence"],
                "search_history": parsed["search_history"],
                "answer": parsed["answer"],
                "clinical_reasoning": parsed["clinical_reasoning"],
                "summary_patient_timeline": previous_timelines[i] if summarize_timeline_for_context else "",
            })

            # Update timeline for next iteration: short version (no VALUE) for model prompt,
            # full version for logging
            full_timeline = format_retrieved_events(all_results_per_patient[i], exclude_report=False)
            short_timeline = format_retrieved_events(
                all_results_per_patient[i], exclude_report=False, exclude_value=False
            )
            prev_timeline_per_patient[i] = short_timeline
            timeline_per_iteration_per_patient[i].append(full_timeline)
            timeline_per_iteration_summarized_per_patient[i].append(full_timeline)

    # Build results (exclude_report=False to include all BM25 hits in timeline)
    return [
        IterativeRetrievalResult(
            timeline_str=format_retrieved_events(all_results_per_patient[i], exclude_report=False),
            timeline_per_iteration=timeline_per_iteration_per_patient[i],
            timeline_per_iteration_summarized=timeline_per_iteration_summarized_per_patient[i],
            iterations_log=iterations_log_per_patient[i],
            all_keywords=list(dict.fromkeys(all_keywords_flat_per_patient[i])),
            keyword_reasoning=keyword_reasoning_per_patient[i],
        )
        for i in range(n)
    ]
