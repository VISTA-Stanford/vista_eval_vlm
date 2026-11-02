"""Qwen3-VL 32B Instruct, through vLLM on weill.

Qwen publishes no CT or medical recipe (model card, github.com/QwenLM/Qwen3-VL README and the tech report, read
2026-11-01). Spec: planning/vista_eval_vlm/2026-11-01-open-vlm-fair-eval.md (on the mount). Taken from that search:
  * the input is Astra's: 50 uniform axial slices for one scan, 25 + 25 for two, native 512 px lossless PNG in
    MedGemma's three windows, so it reads the same image packs as Astra. Qwen-family CT papers feed 24-64 separate
    slices; separate images, not video (the video path adds timestamps that mean nothing for slices). At 512 px a
    slice is 16 x 16 = 256 tokens (32-px merged patches), inside the processor's default pixel range, so it is not
    resized;
  * the prompt is the v1 plain card prompt: each image followed by "SLICE n", the question, then the
    "Final Answer: yes/no" line; no persona and no system message (the chat template's default);
  * greedy decoding, up to 2,000 new tokens, as the README's own vLLM example (temperature 0); yes/no token
    log-probabilities logged (top 20), not scored.
"""

from .base import Recipe

RECIPE = Recipe(
    name="qwen3vl",
    model_type="vllm",
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    n_slices=50,
    n_slices_pair=25,
    sampler="medgemma_uniform",
    windows="medgemma",
    size=None,
    image_format="png",
    jpeg_quality=None,
    label_style="after",
    label_suffix="\n",
    answer_form="final_answer",
    answer_line="Answer with 'Final Answer: yes' or 'Final Answer: no'.",
    # 65,536 context: a few scans store 1,024-px slices (1,024 tokens each, ~52k for 50), answered at native pixels as
    # Astra was sent them; 512-px scans use ~13k either way.
    settings=(("temperature", 0.0), ("max_new_tokens", 2000), ("logprobs", 20), ("max_model_len", 65536),
              ("no_video", True), ("max_image_aspect", 200)),  # the processor's limit (smart_resize)
)
