"""OctoMed 7B (Qwen2.5-VL-7B base), through vLLM on weill.

Sources (read 2026-11-01): the model card huggingface.co/OctoMed/OctoMed-7B and the paper arXiv 2511.23269. Neither
evaluates CT, volumes or more than 10 images. Spec: planning/vista_eval_vlm/2026-11-01-open-vlm-fair-eval.md (mount).
Taken from them:
  * a fixed 512 x 512 pixel budget (min_pixels = max_pixels = 262,144; ~361 tokens a slice after Qwen2.5-VL's
    28-px rounding), and at most 10 images, the most the paper evaluated with: 10 uniform axial slices for one
    scan, 5 + 5 for two. Pixels are Astra's (MedGemma's three windows, native 512 px, PNG);
  * the card's multiple-choice form: the question, the options one per line ("A: Yes", "B: No"), then "Please
    reason step-by-step, and put your final answer within \\boxed{}."; the default system prompt ("You are a helpful
    assistant.", which the card warns it is sensitive to); no persona. It reasons in <think> before the answer;
  * the card's sampling: temperature 0.6, top_p 0.95, up to 8,192 new tokens; one sample with a fixed seed.
Ours, not the card's: "SLICE n" after each image (as for the other open models); the option-letter
log-probabilities logged (A as yes, B as no), never scored.
"""

from .base import Recipe

RECIPE = Recipe(
    name="octomed",
    model_type="vllm",
    model_name="OctoMed/OctoMed-7B",
    n_slices=10,
    n_slices_pair=5,
    sampler="medgemma_uniform",
    windows="medgemma",
    size=None,
    image_format="png",
    jpeg_quality=None,
    label_style="after",
    label_suffix="\n",
    answer_form="boxed_choice",
    answer_line="A: Yes\nB: No\nPlease reason step-by-step, and put your final answer within \\boxed{}.",
    system="You are a helpful assistant.",
    settings=(("temperature", 0.6), ("top_p", 0.95), ("seed", 20261101), ("max_new_tokens", 8192),
              ("logprobs", 20), ("max_model_len", 16384), ("no_video", True),
              ("mm_processor_kwargs", (("min_pixels", 262144), ("max_pixels", 262144)))),
)
