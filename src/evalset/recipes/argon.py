"""Gemini 4 Argon on Vertex AI (location global), run from phil-sllm-01.

From the per-model guidance (planning/vista_eval_vlm/session/2026-10-26-per-model-input-guidance.md)
and Google's media-resolution and image-understanding pages:
  * separate images, each labelled ("Image n:"), the question after them (images first, so one
    explicit context cache per scan holds them and each question is a short call on top);
  * media resolution set to high explicitly (Google's recommendation for images with small detail);
  * a response schema forces "yes" or "no"; temperature is ignored by the model, thinking "low";
  * 50 slices for one scan, 25 + 25 for two: the same counts as Astra, the common ceiling.
Windows: MedGemma's set (guidance, cross-model points). Lossless PNG at native 512 px (Phil 10-28): JPEG's colour
sub-sampling blurred the three windows by ~5-7 grey levels; a 50-slice PNG request (22.0 MB) was accepted on 10-28,
55,019 prompt tokens, the same as JPEG.

No explicit context cache: Vertex answers "Model `gemini-4-argon` does not support explicit context
caching" (HTTP 400, measured 2026-10-28, $0; Google's docs: not supported "for Gemini 4 Argon and onward").
Google's implicit cache does work once the first question on a scan has finished: questions 2-4 then read
~53k of 55k prompt tokens from cache (measured 10-28, $0.14 for 4 calls). The adapter sends each scan's
first question alone, then the rest (``VertexGeminiAdapter.ask_group``). Sent all at once, none hit.
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
    image_format="png",
    jpeg_quality=None,
    label_style="before",
    answer_form="schema",
    settings=(("media_resolution", "MEDIA_RESOLUTION_HIGH"), ("thinking_level", "low"),
              ("max_new_tokens", 8192), ("explicit_cache", False)),
)
