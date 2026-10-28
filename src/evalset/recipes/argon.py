"""Gemini 4 Argon on Vertex AI (location global), run from phil-sllm-01.

From the per-model guidance (planning/vista_eval_vlm/session/2026-10-26-per-model-input-guidance.md)
and Google's media-resolution and image-understanding pages:
  * separate images, each labelled ("Image n:"), the question after them (images first, so one
    explicit context cache per scan holds them and each question is a short call on top);
  * media resolution set to high explicitly (Google's recommendation for images with small detail);
  * a response schema forces "yes" or "no"; temperature is ignored by the model, thinking "low";
  * 50 slices for one scan, 25 + 25 for two: the same counts as Astra, the common ceiling.
Windows: MedGemma's set (guidance, cross-model points). JPEG at quality 95.

No explicit context cache: Vertex answers "Model `gemini-4-argon` does not support explicit context
caching" (HTTP 400, measured 2026-10-28 from phil-sllm-01, $0), and implicit caching did not fire on
10-27. Each question is a full call. The adapter's cache path stays for models that support it.
"""

from .base import Recipe

RECIPE = Recipe(
    name="argon",
    model_type="vertex_gemini",
    model_name="gemini-4-argon",
    n_slices=50,
    n_slices_pair=25,
    windows="medgemma",
    size=None,
    image_format="jpeg",
    jpeg_quality=95,
    label_style="before",
    answer_form="schema",
    settings=(("media_resolution", "MEDIA_RESOLUTION_HIGH"), ("thinking_level", "low"),
              ("max_new_tokens", 8192), ("explicit_cache", False)),
)
