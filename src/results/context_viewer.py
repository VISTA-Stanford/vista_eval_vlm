"""Weight-free config-context viewer (Phase 2).

Renders **exactly** the context a given ``(config, task, experiment, model)`` feeds
each VLM — the assembled prompt text + the selected CT slices / pathology tiles
(post-window, model-ready) as base64 thumbnails, per example, paginated by batch,
with a text-token-budget bar — into **one self-contained HTML file**.

The concrete use is Phil's **manual pre-run input QA**: before kicking off a full
eval run, eyeball a *small subset* of examples across a couple of VLM tasks to
confirm the inputs look correct — prompt text well-formed, right slice counts, no
0-image rows where images are expected. It *instruments* the preprocessing
pipeline ("you don't trust; you instrument") rather than scoring it — it shows
**inputs, not answers**. There is no correctness coloring and no ``--results-csv``
(Q3): the viewer has no model response to score against, by design.

ZERO DRIFT WITH INFERENCE
-------------------------
The viewer reuses the *same* weight-free capture core as the golden harness
(``vista_run.context_capture.iter_captured_examples``): build the orchestrator
WITHOUT weights, ``_load_experiment_data`` -> ``_build_prompts_for_experiment`` ->
iterate ``PromptDataset.__getitem__`` directly. ``CapturedExample.images`` is the
fully-preprocessed PIL image(s) the model actually receives — the golden hashes
them, the viewer base64-thumbnails them. One capture path => what you see is what
the model gets.

WEIGHT-FREE BUT VM-ONLY / PHI
-----------------------------
``TaskOrchestrator.__init__`` builds the BigQuery + GCS clients and the data path
reads real de-identified PHI, so this runs **on the VM only** (Claude-Code CPU box),
like the golden harness — no GPU, no model weights. The emitted HTML embeds real
de-identified timelines + imagery (PHI): it is written only under the config's
``results_dir`` (su-vista mount, outside the repo tree), is git-ignored
(``*_context_view.html``), and must never be committed. Readback to the planner Mac
reports counts / file existence / slice counts / token numbers only.

USAGE
-----
    cd src && python -m results.context_viewer \
        --config configs/all_tasks.yaml \
        --type gemma3 --name google/medgemma-1.5-4b-it \
        --task progression_recurrence_free_survival_1_yr \
        --experiment axial_all_image \
        --limit 5

``--type`` / ``--name`` mirror Ryan's repo-wide convention (``run_bq.py`` and every
eval script): the ``MODEL_REGISTRY`` key + the model HF name. ``--experiment`` is
SINGULAR (the viewer renders one experiment) and is resolved through
``normalize_experiments(cfg)`` — it must be one of the config's experiments.
``--limit`` caps rows AFTER the ``(person_id, index)`` sort (keep it small — this is
pre-run QA on a subset). ``--batches`` caps the number of display pages emitted.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

from vista_run.run_bq import TaskOrchestrator, RETRIEVAL_EXPERIMENTS
from vista_run.context_capture import iter_captured_examples
from context.normalize import normalize_experiments
from context.selectors.ehr_filters import MODEL_BACKED_FILTERS
from models import MODEL_REGISTRY

# Text-token budget color thresholds (fraction of context_window).
_BUDGET_GREEN = 0.70
_BUDGET_AMBER = 0.90

# Row metadata columns worth surfacing as a "label" chip, in preference order.
# Best-effort: the first one present on the row is shown (input QA context, not a
# score — the viewer never has the model's answer).
_LABEL_COLUMNS = ("label", "answer", "correct_answer", "target", "y_true")

# Thumbnail box (px) for the HTML — the captured PILs are already model-ready
# (448/512); we downscale a display copy so the self-contained HTML stays small.
_THUMB_PX = 160


class TokenizerLoadError(RuntimeError):
    """Raised when the standalone tokenizer-only load fails (Q1: hard STOP).

    A miss on these VMs is *diagnosable*, not random: ``TaskOrchestrator._set_envs``
    redirects ``HF_HOME`` to ``cfg['runtime']['cache_dir']`` (so a tokenizer in the
    default ``~/.cache/huggingface`` looks missing, and the same redirect hides the
    ``hf auth login`` token -> gated-repo 401 for gemma/medgemma); the model's
    tokenizer may never have been downloaded on this box; or the VM may be offline.
    The message names the model + likely causes so Phil can pick the remedy when it
    fires ("decide when we get to it").
    """

    def __init__(self, model_name, cache_dir, errors):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.errors = errors
        detail = "\n  - ".join(errors) if errors else "(no specific error captured)"
        super().__init__(
            f"STOP: could not load a tokenizer for model '{model_name}' "
            f"(cache_dir={cache_dir}).\n"
            f"Attempts:\n  - {detail}\n"
            "Likely cause (pick the remedy):\n"
            "  * wrong cache_dir — the tokenizer sits in a different HF cache than "
            "runtime.cache_dir;\n"
            "  * gated repo (gemma/medgemma) needs HF_TOKEN forwarded (the HF_HOME "
            "redirect hides `hf auth login`);\n"
            "  * this model's tokenizer was never downloaded on this box;\n"
            "  * the VM is offline (no network to fetch it).\n"
            "The viewer will not fabricate a token bar from a whitespace split — a "
            "fake budget would undermine the QA surface. Stage the tokenizer / forward "
            "the token, or relax to an estimate (Phil's call)."
        )


# --------------------------------------------------------------------------- #
# Tokenizer-only counter (weight-free)                                         #
# --------------------------------------------------------------------------- #
def load_tokenizer_only(model_name, cache_dir):
    """Load ONLY the tokenizer (no GPU, no model weights).

    ``run_bq.TaskOrchestrator._count_prompt_tokens`` reads the already-loaded
    ``self.processor``/``self.model`` tokenizer -> weight-bearing, unusable here.
    We load a standalone tokenizer honoring the run's ``cache_dir`` (the orchestrator
    already redirected ``HF_HOME`` to it via ``_set_envs``): try ``AutoProcessor`` ->
    its ``.tokenizer`` first, then ``AutoTokenizer``. On failure -> ``TokenizerLoadError``
    (hard STOP; Q1).
    """
    from transformers import AutoProcessor, AutoTokenizer

    errors = []
    try:
        proc = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        tok = getattr(proc, "tokenizer", None)
        if tok is not None and hasattr(tok, "encode"):
            return tok
        errors.append("AutoProcessor loaded but exposed no .tokenizer with .encode")
    except Exception as e:  # noqa: BLE001 — surface the concrete cause in the STOP
        errors.append(f"AutoProcessor: {type(e).__name__}: {e}")
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        if hasattr(tok, "encode"):
            return tok
        errors.append("AutoTokenizer loaded but has no .encode")
    except Exception as e:  # noqa: BLE001
        errors.append(f"AutoTokenizer: {type(e).__name__}: {e}")
    raise TokenizerLoadError(model_name, cache_dir, errors)


def count_text_tokens(tokenizer, text):
    """Exact text-token count for ``text`` (``add_special_tokens=False``)."""
    if text is None:
        return 0
    text = str(text)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        # A rare tokenizer whose encode() lacks the kwarg.
        return len(tokenizer.encode(text))


# --------------------------------------------------------------------------- #
# Experiment resolution + fail-closed preflight                               #
# --------------------------------------------------------------------------- #
def resolve_experiment(cfg, experiment):
    """Resolve ``--experiment`` through ``normalize_experiments(cfg)``.

    Returns the single matching ``NormalizedExperiment`` (its ``.name`` is the
    legacy/normalized token the capture core consumes). Raises ``SystemExit`` for an
    unknown name (no match) or an ambiguous one (>1 match) — fail-closed, before any
    data load or weights path.
    """
    normalized = normalize_experiments(cfg)
    matches = [n for n in normalized if n.name == experiment]
    if not matches:
        available = ", ".join(sorted({n.name for n in normalized})) or "(none)"
        raise SystemExit(
            f"STOP: unknown --experiment '{experiment}': no match in the config's "
            f"experiments. Available (normalized) names: {available}. "
            "The viewer renders an experiment declared in the config."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"STOP: ambiguous --experiment '{experiment}': {len(matches)} config "
            "entries normalize to this name. De-duplicate the config's experiments."
        )
    return matches[0]


def _uses_model_backed_summarize(norm):
    """True if the resolved experiment uses a model-backed filter step (e.g. summarize).

    Fail-closed (like ``assembler.preflight`` fail-closing ``inline_by_timestamp``):
    a model-backed step is not weight-free; a viewer that silently rendered it would
    misrepresent the context. Detected three ways, most authoritative first:

      1. any block's ``config['select']`` chain (a single ``{fn: ...}`` dict OR a
         list of them) names a filter in ``MODEL_BACKED_FILTERS`` — the canonical
         seam in ``context.selectors.ehr_filters`` and the form the EHR ``select``
         chain actually uses (e.g. ``{"fn": "summarize"}``). Sourcing the set from
         there means a *future* model-backed filter fail-closes with no viewer edit;
      2. a truthy ``summarize`` / ``summarization`` key directly on a block config;
      3. the experiment name contains ``summariz`` (covers the legacy
         ``*_summarization*`` retrieval presets, already retrieval-rejected too).
    """
    name = (norm.name or "").lower()
    if "summariz" in name:
        return True
    for block in (norm.blocks or []):
        if not isinstance(block, dict):
            continue
        block_cfg = block.get("config", {})
        if not isinstance(block_cfg, dict):
            continue
        if block_cfg.get("summarize") or block_cfg.get("summarization"):
            return True
        select = block_cfg.get("select")
        # `select` may be a single selector dict or a list/tuple of them.
        specs = select if isinstance(select, (list, tuple)) else [select]
        for spec in specs:
            if isinstance(spec, dict) and spec.get("fn") in MODEL_BACKED_FILTERS:
                return True
    return False


def _experiment_expects_images(norm):
    """True if this experiment's blocks include a CT / pathology modality.

    Used only for the input-sanity flag (a 0-image row where images ARE expected);
    never for correctness. A text-only experiment (e.g. ``no_image``, ``timeline_only``)
    legitimately has 0 images and is not flagged.
    """
    for block in (norm.blocks or []):
        if not isinstance(block, dict):
            continue
        if block.get("modality") in ("volume3d", "patches2d"):
            return True
        if block.get("adapter") in ("ct", "pathology"):
            return True
    return False


# --------------------------------------------------------------------------- #
# HTML rendering (self-contained: inline CSS, base64 images, no external refs) #
# --------------------------------------------------------------------------- #
def _esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def _image_to_data_uri(img):
    """Downscaled base64 PNG data-URI for one PIL image (self-contained; no path).

    Never emits a filesystem path in the src — a copy is thumbnailed and inlined so
    the HTML carries the pixels, satisfying the "no local image path in <img src>"
    self-containment gate.
    """
    thumb = img.copy()
    thumb.thumbnail((_THUMB_PX, _THUMB_PX))
    if thumb.mode not in ("RGB", "L"):
        thumb = thumb.convert("RGB")
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _images_html(images):
    """Thumbnail grid for item['image'] (None | PIL | list-of-PIL), in assembly order."""
    if images is None:
        return '<div class="noimg">0 images (no_image row)</div>'
    image_list = images if isinstance(images, list) else [images]
    if not image_list:
        return '<div class="noimg">0 images (no_image row)</div>'
    cells = []
    for pos, img in enumerate(image_list):
        try:
            uri = _image_to_data_uri(img)
            cells.append(f'<figure><img src="{uri}" alt="slice {pos}"/><figcaption>{pos}</figcaption></figure>')
        except Exception as e:  # noqa: BLE001 — one bad thumbnail must not sink the page
            cells.append(f'<figure class="bad">thumb err: {_esc(type(e).__name__)}</figure>')
    return f'<div class="thumbs">{"".join(cells)}</div>'


def _token_bar_html(text_tokens, context_window):
    """Text-token budget bar. ``context_window is None`` -> 'unknown' (no false fill)."""
    if context_window is None:
        return (
            '<div class="tokbar unknown">'
            f'<span class="toklabel">text tokens: {text_tokens} / '
            'context window unknown</span></div>'
        )
    pct = text_tokens / context_window if context_window else 0.0
    if pct < _BUDGET_GREEN:
        cls = "green"
    elif pct < _BUDGET_AMBER:
        cls = "amber"
    else:
        cls = "red"
    width = min(100.0, max(0.0, pct * 100.0))
    return (
        f'<div class="tokbar {cls}">'
        f'<div class="fill" style="width:{width:.1f}%"></div>'
        f'<span class="toklabel">text tokens: {text_tokens} / {context_window} '
        f'({pct * 100:.1f}%)</span></div>'
    )


def _selected_preview(selected_indices):
    """Compact 'selected: N [preview]' chip content for CT slice indices / path tiles."""
    if selected_indices is None:
        return "selected: —"
    try:
        n = len(selected_indices)
    except TypeError:
        return f"selected: {_esc(selected_indices)}"
    preview = ", ".join(str(s) for s in list(selected_indices)[:8])
    if n > 8:
        preview += ", …"
    # For pathology tiles selected_indices are file paths; show basenames to keep the
    # chip readable (the HTML is PHI-mount-only, but a full path adds no QA value).
    if selected_indices and isinstance(selected_indices[0], str):
        base = ", ".join(Path(str(s)).name for s in list(selected_indices)[:6])
        if n > 6:
            base += ", …"
        return f"selected: {n} [{_esc(base)}]"
    return f"selected: {n} [{_esc(preview)}]"


def _label_chip(raw_row):
    for col in _LABEL_COLUMNS:
        try:
            val = raw_row.get(col)
        except Exception:  # noqa: BLE001
            val = None
        if val is not None and str(val) != "nan":
            return f"{_esc(col)}: {_esc(val)}"
    return None


def _card_html(ex, text_tokens, context_window, expects_images):
    chips = [
        f'<span class="chip">index: {_esc(ex.index)}</span>',
        f'<span class="chip">person_id: {_esc(ex.person_id)}</span>',
        f'<span class="chip">task: {_esc(ex.task)}</span>',
        f'<span class="chip">images: {_esc(ex.image_count)}</span>',
        f'<span class="chip">{_esc(_selected_preview(ex.selected_indices))}</span>',
        f'<span class="chip">assembly: {_esc(ex.assembly_mode)}</span>',
        f'<span class="chip">windowing: {_esc(ex.windowing)}</span>',
    ]
    label = _label_chip(ex.raw_row)
    if label is not None:
        chips.insert(3, f'<span class="chip label">{label}</span>')
    if ex.path_tile_count is not None:
        chips.append(f'<span class="chip">path_tiles: {_esc(ex.path_tile_count)}</span>')

    # Input-sanity flags ONLY (never correctness): a 0-image row where images are
    # expected, or an empty prompt. These are QA red-flags on the inputs themselves.
    warns = []
    if expects_images and ex.image_count == 0:
        warns.append("0 images where images are expected")
    if not (ex.dynamic_prompt or "").strip():
        warns.append("empty prompt")
    warn_html = ""
    if warns:
        warn_html = '<div class="warn">⚠ ' + "; ".join(_esc(w) for w in warns) + "</div>"

    return f"""    <article class="card">
      <div class="chips">{"".join(chips)}</div>
      {warn_html}
      {_token_bar_html(text_tokens, context_window)}
      {_images_html(ex.images)}
      <details class="prompt">
        <summary>assembled prompt ({text_tokens} text tokens)</summary>
        <pre>{_esc(ex.dynamic_prompt)}</pre>
      </details>
    </article>"""


_STYLE = """
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.2rem;
         background: #fafafa; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) { body { background: #16181c; color: #e6e6e6; } }
  h1 { font-size: 1.25rem; margin: 0 0 .2rem; }
  .meta { font-size: .82rem; opacity: .8; margin-bottom: 1rem; line-height: 1.5; }
  .meta code { background: rgba(127,127,127,.18); padding: .05rem .3rem; border-radius: 3px; }
  .caveat { font-size: .78rem; opacity: .75; border-left: 3px solid #b8860b;
            padding: .3rem .6rem; margin: .6rem 0; }
  .batch { margin: 1.4rem 0 .4rem; font-weight: 600; font-size: .95rem;
           border-bottom: 1px solid rgba(127,127,127,.35); padding-bottom: .2rem; }
  .card { border: 1px solid rgba(127,127,127,.3); border-radius: 8px; padding: .8rem;
          margin: .7rem 0; background: rgba(127,127,127,.05); }
  .chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .5rem; }
  .chip { font-size: .74rem; background: rgba(127,127,127,.16); border-radius: 10px;
          padding: .12rem .55rem; white-space: nowrap; }
  .chip.label { background: rgba(56,142,60,.22); }
  .warn { color: #c0392b; font-size: .82rem; font-weight: 600; margin: .3rem 0; }
  .tokbar { position: relative; height: 1.25rem; border-radius: 4px; overflow: hidden;
            background: rgba(127,127,127,.2); margin: .4rem 0; }
  .tokbar .fill { position: absolute; inset: 0 auto 0 0; height: 100%; }
  .tokbar.green .fill { background: #4caf50; }
  .tokbar.amber .fill { background: #ffb300; }
  .tokbar.red .fill { background: #e53935; }
  .tokbar.unknown { background: repeating-linear-gradient(45deg,
            rgba(127,127,127,.15), rgba(127,127,127,.15) 6px,
            rgba(127,127,127,.28) 6px, rgba(127,127,127,.28) 12px); }
  .toklabel { position: relative; font-size: .72rem; line-height: 1.25rem;
              padding-left: .4rem; mix-blend-mode: difference; color: #fff; }
  .thumbs { display: flex; flex-wrap: wrap; gap: .3rem; margin: .4rem 0; }
  .thumbs figure { margin: 0; text-align: center; }
  .thumbs img { max-width: 160px; height: auto; border: 1px solid rgba(127,127,127,.3);
                border-radius: 3px; display: block; }
  .thumbs figcaption { font-size: .65rem; opacity: .7; }
  .thumbs figure.bad { font-size: .7rem; color: #c0392b; }
  .noimg { font-size: .8rem; opacity: .7; font-style: italic; margin: .3rem 0; }
  details.prompt { margin-top: .4rem; }
  details.prompt summary { cursor: pointer; font-size: .8rem; opacity: .85; }
  details.prompt pre { white-space: pre-wrap; word-break: break-word; font-size: .78rem;
            background: rgba(127,127,127,.1); padding: .6rem; border-radius: 5px;
            max-height: 30rem; overflow: auto; }
"""


def render_html(orch, norm, examples, tokenizer, context_window, batch_size, max_batches=None):
    """Assemble the full self-contained HTML document from captured examples.

    Paginated into display pages of ``batch_size`` rows. NOTE the pages are a display
    grouping — they are NOT the runtime's exact batching (Gemma groups a batch by
    image count for inference efficiency; that boundary is cosmetic for input QA).
    """
    expects_images = _experiment_expects_images(norm)
    cw_label = "unknown" if context_window is None else str(context_window)

    header = f"""  <h1>Config-context viewer — {_esc(norm.name)}</h1>
  <div class="meta">
    model: <code>{_esc(orch.model_type)}</code> / <code>{_esc(orch.model_name)}</code>
    &nbsp;·&nbsp; task: <code>{_esc(examples[0].task if examples else '?')}</code>
    &nbsp;·&nbsp; experiment: <code>{_esc(norm.name)}</code>
    &nbsp;·&nbsp; assembly: <code>{_esc(norm.assembly)}</code>
    &nbsp;·&nbsp; context_window: <code>{_esc(cw_label)}</code>
    &nbsp;·&nbsp; examples: <code>{len(examples)}</code>
  </div>
  <div class="caveat">Input QA only — this shows what the model <em>receives</em>, not
    what it answers (no correctness scoring, by design). The token bar is a
    <strong>text-token</strong> budget; VLM image tokens are model-specific and are
    <strong>not</strong> included (image count is shown separately per card).</div>
  <div class="caveat">Pages below are display groups of {batch_size} rows; the runtime's
    Gemma image-count batching may draw batch boundaries differently.</div>"""

    body_parts = [header]
    n = len(examples)
    page = 0
    for start in range(0, n, batch_size):
        if max_batches is not None and page >= max_batches:
            body_parts.append(
                f'  <div class="caveat">… {n - start} further example(s) omitted '
                f'(--batches {max_batches}).</div>'
            )
            break
        page += 1
        chunk = examples[start:start + batch_size]
        body_parts.append(
            f'  <div class="batch">Page {page} — rows {start}–{start + len(chunk) - 1}</div>'
        )
        for ex in chunk:
            text_tokens = count_text_tokens(tokenizer, ex.dynamic_prompt)
            body_parts.append(_card_html(ex, text_tokens, context_window, expects_images))

    body = "\n".join(body_parts)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Config-context viewer — {_esc(norm.name)}</title>
<style>{_STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""


def default_out_path(orch, task_info, experiment):
    """HTML output path under the config results_dir (PHI mount, git-ignored).

    Mirrors ``golden_harness.default_out_path`` layout so viewer output lands beside
    the golden dumps under the su-vista mount and is caught by the ``*_context_view.html``
    / ``context_view/`` .gitignore backstop.
    """
    src = task_info.get("task_source_csv")
    parts = [orch.results_base, "context_view"]
    if src:
        parts.append(src)
    parts.extend([task_info["task_name"], orch.file_model_name])
    out_dir = Path(*[str(p) for p in parts])
    return out_dir / f"{task_info['task_name']}_{experiment}_context_view.html"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--type", required=True, help="model_type (MODEL_REGISTRY key), e.g. gemma3")
    parser.add_argument("--name", required=True, help="model HF path, e.g. google/medgemma-1.5-4b-it")
    parser.add_argument("--task", required=True, help="single task_name to render")
    parser.add_argument("--experiment", required=True,
                        help="single experiment (must be one of the config's experiments)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows AFTER the (person_id, index) sort (keep small — pre-run QA on a subset)")
    parser.add_argument("--batches", type=int, default=None,
                        help="cap the number of display pages emitted")
    parser.add_argument("--out", default=None, help="override output HTML path (must end with _context_view.html)")
    args = parser.parse_args()

    # Preflight (iv) — invalid --type, before building BQ/GCS clients.
    if args.type not in MODEL_REGISTRY:
        raise SystemExit(
            f"STOP: unknown --type '{args.type}'. Valid model types: "
            f"{', '.join(sorted(MODEL_REGISTRY))}."
        )

    # --out PHI backstop: keep the git-ignored suffix wherever the file lands.
    if args.out is not None and not str(args.out).endswith("_context_view.html"):
        raise SystemExit(
            "STOP: --out must end with '_context_view.html' so the PHI .gitignore "
            "backstop applies (the HTML embeds de-identified timelines / imagery)."
        )

    # Build the orchestrator WEIGHT-FREE (constructs BQ/GCS clients + the adapter;
    # never loads model weights — VM-only). Assert the weights path is untouched.
    orch = TaskOrchestrator(args.config, args.type, args.name)
    assert orch.model is None and orch.processor is None, (
        "context_viewer must stay weight-free: orchestrator model/processor loaded"
    )

    task_info = next((t for t in orch.valid_tasks if t["task_name"] == args.task), None)
    if task_info is None:
        raise SystemExit(f"STOP: task '{args.task}' not found in valid_tasks ({args.config}).")

    # Preflight (iii) — resolve/validate the experiment (unknown / ambiguous -> STOP).
    norm = resolve_experiment(orch.cfg, args.experiment)

    # Preflight (i) — retrieval experiments need model weights.
    if norm.name in RETRIEVAL_EXPERIMENTS or getattr(norm, "legacy_retrieval", False):
        raise SystemExit(
            f"STOP: experiment '{norm.name}' is a retrieval experiment; its prompt "
            "build requires model weights and is out of scope for the weight-free viewer."
        )

    # Preflight (ii) — model-backed summarize step is not weight-free (fail-closed).
    if _uses_model_backed_summarize(norm):
        raise SystemExit(
            f"STOP: experiment '{norm.name}' uses a model-backed summarize step, which "
            "is not weight-free. The viewer fail-closes rather than misrepresent the "
            "context by silently skipping it."
        )

    # Preflight (v) — the standalone tokenizer-only load MUST succeed (Q1: hard STOP).
    cache_dir = orch.cfg.get("runtime", {}).get("cache_dir")
    tokenizer = load_tokenizer_only(args.name, cache_dir)  # raises TokenizerLoadError -> STOP

    # context_window off the constructed (weight-free) adapter; None -> "unknown" bar.
    context_window = getattr(orch.adapter, "context_window", None)

    # Capture (weight-free) + render.
    examples = list(iter_captured_examples(orch, task_info, norm.name, orch.model_type, limit=args.limit))
    if not examples:
        raise SystemExit(
            f"STOP: no examples captured for task '{args.task}' / experiment "
            f"'{norm.name}' (empty cohort or all rows dropped)."
        )

    batch_size = orch.cfg.get("runtime", {}).get("batch_size", 64)
    document = render_html(
        orch, norm, examples, tokenizer, context_window, batch_size, max_batches=args.batches
    )

    out_path = Path(args.out) if args.out else default_out_path(orch, task_info, norm.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")

    rendered = len(examples) if args.batches is None else min(len(examples), args.batches * batch_size)
    cw_label = "unknown" if context_window is None else context_window
    print(f">>> Wrote config-context view -> {out_path}")
    print(f"    examples={len(examples)} rendered={rendered} context_window={cw_label} "
          f"batch_size={batch_size}")


if __name__ == "__main__":
    main()
