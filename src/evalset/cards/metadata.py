"""Step 5: positive control, two questions any model that sees the scan should get right (one call per item).

Not part of the 21-question eval set: a check that a model's images arrive intact before its eval-set scores are
read (spec planning/vista_eval_vlm/2026-10-30-medgemma-fair-eval.md). Answer keys come from each series' DICOM
header (``samples.step5``): IV contrast where the study description and the contrast-agent field agree, and the
patient's sex. Both are visible in the images: contrast in the soft-tissue window, sex from the chest wall and breasts.
"""

from .base import ANSWER, Card, Question

CARD = Card(
    step=5,
    title="Positive control: IV contrast and sex (one call per item)",
    opening="These are axial slices of a chest CT, ordered from top (head) to bottom (feet), each shown in {windows}.",
    joiner="\n\n",
    questions=(
        Question("meta_iv_contrast", "meta_iv_contrast",
                 f"Question: Was this scan taken with intravenous (IV) contrast?\n{ANSWER}"),
        Question("meta_female", "meta_female", f"Question: Is this patient female?\n{ANSWER}"),
    ),
)
