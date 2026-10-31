"""Claude Opus 5.5 on Vertex AI (location global), run from phil-sllm-01 like Argon.

Same pixels and wrapper as Argon and Astra, so the bars compare models, not inputs:
  * 50 slices for one scan, 25 + 25 for two; MedGemma's windows; lossless PNG at native 512 px;
  * separate images, each labelled ("Image n:"), the question after them;
  * structured output forces {"answer": "yes" | "no"} (Opus 5.5 rejects forced tool choice and temperature);
  * effort "low": Opus 5.5's thinking cannot be turned off, so this is the lightest setting (Argon runs thinking low);
  * the prompt cache marks the last image, so a scan's later questions read the images at the cache-read rate.
Measured 10-26: ~370 prompt tokens per 512 px slice (50 slices ~18.5k tokens, a third of Argon's ~55k).
No agent and no tools: one call per question (Phil 10-31).
"""

from .base import Recipe

RECIPE = Recipe(
    name="opus",
    model_type="vertex_claude",
    model_name="claude-opus-5-5",
    n_slices=50,
    n_slices_pair=25,
    windows="medgemma",
    size=None,
    image_format="png",
    jpeg_quality=None,
    label_style="before",
    answer_form="schema",
    settings=(("effort", "low"), ("max_new_tokens", 4096), ("concurrency", 8), ("prompt_cache", True)),
)
