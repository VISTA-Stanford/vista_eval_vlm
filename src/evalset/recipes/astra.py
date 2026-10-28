"""GPT-6 Astra on the Stanford Health Care AI Hub (Azure), run from weill.

From the per-model guidance (planning/vista_eval_vlm/session/2026-10-26-per-model-input-guidance.md):
  * at most 50 images per call (the gateway's measured cap): 50 slices for one scan, 25 + 25 for two;
  * images before the question, each labelled ("Image n:"), per OpenAI's vision guidance;
  * a strict JSON schema forces {"answer": "yes" | "no"}; no temperature or logprobs (refused);
  * reasoning effort "low", the cheapest legal setting (the pilot compares "medium");
  * native 512 px (896 px costs 3x the tokens; the pilot can test it).
Windows: MedGemma's set, so every model sees the same pixels (guidance, cross-model points).
JPEG at quality 95, as the AI Hub adapter already sends.
"""

from .base import Recipe

RECIPE = Recipe(
    name="astra",
    model_type="aihub_openai",
    model_name="gpt-6-astra",
    n_slices=50,
    n_slices_pair=25,
    windows="medgemma",
    size=None,
    image_format="jpeg",
    jpeg_quality=95,
    label_style="before",
    answer_form="schema",
    max_images=50,
    settings=(("reasoning_effort", "low"), ("max_new_tokens", 4096), ("concurrency", 4)),
)
