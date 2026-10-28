"""MedGemma 1.5 4B: exactly Google's published CT recipe.

Sources (opened 2026-10-27): the official notebook
github.com/Google-Health/medgemma notebooks/high_dimensional_ct_hugging_face.ipynb and the tech report
arxiv.org/abs/2604.05081 (Table 13, the chest-CT CT-RATE prompt). Taken from them:
  * up to 85 axial slices, ``round(i / 85 * (n - 1))`` for i = 1..85 (``slices.medgemma_uniform``);
  * RGB windows wide -1024..1024 / soft tissue -135..215 / brain 0..80 HU, rounded to uint8;
  * slices encoded as JPEG at PIL's default quality, not resized (the processor makes them 896 px);
  * each image followed by the text "SLICE n" (n counts from 1), the question after all images;
  * "You are an expert radiologist for chest CT." and the answer line
    "Answer with 'Final Answer: yes' or 'Final Answer: no'." (Table 13, CT-RATE);
  * greedy decoding, up to 2,000 new tokens (the notebook's generate call).
Ours, not Google's: two-scan cards get 42 slices per scan (84 in all, inside the 85 it was trained
and evaluated on), labelled "PRIOR SLICE n" / "CURRENT SLICE n"; the yes/no token probabilities are
logged (top 20 per position), never scored. Spec: per-model guidance, session/2026-10-26.
"""

from .base import Recipe

RECIPE = Recipe(
    name="medgemma",
    model_type="vllm",
    model_name="google/medgemma-1.5-4b-it",
    n_slices=85,
    n_slices_pair=42,
    sampler="medgemma_uniform",
    windows="medgemma",
    size=None,
    image_format="jpeg",
    jpeg_quality=None,
    label_style="after",
    answer_form="final_answer",
    persona="You are an expert radiologist for chest CT.",
    answer_line="Answer with 'Final Answer: yes' or 'Final Answer: no'.",
    settings=(("temperature", 0.0), ("max_new_tokens", 2000), ("logprobs", 20), ("max_model_len", 32768)),
)
