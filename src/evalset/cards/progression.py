"""Step 3: radiologic progression vs the prior CT, two-way.

Answer key: vista_bench ``radiology_progression`` collapsed to the approved event: progression or
mixed response (labels 1, 3) = yes; stable or response (0, 2) = no. Prompt collapsed from the
radprog prompt (prompts_by_task.radprog_v1_7.json).
"""

from .base import ANSWER, Card, Question

#: vista_bench radiology_progression label -> the two-way answer (1 = yes).
TWO_WAY = {0: 0, 1: 1, 2: 0, 3: 1}

CARD = Card(
    step=3,
    title="Progression vs the prior CT (two-way)",
    opening=("These are axial slices of two chest CTs of the same patient, a PRIOR scan and the CURRENT scan, "
             "each ordered from top to bottom and shown in {windows}."),
    joiner="\n\n",
    questions=(Question("progression_vs_prior", "radiology_progression",
                        "Question: Comparing the CURRENT scan with the PRIOR scan, has the cancer progressed?\n"
                        "- Yes: progression or mixed response: clear growth of known malignant lesions, or new lesions "
                        "suspicious for cancer, even if other sites improved.\n- No: stable disease or response.\n"
                        "Do not count non-cancer changes (infection, aspiration, organizing pneumonia, drug reaction, "
                        "evolving post-radiation change).\n" + ANSWER),),
    scan_names=("PRIOR", "CURRENT"),
)
