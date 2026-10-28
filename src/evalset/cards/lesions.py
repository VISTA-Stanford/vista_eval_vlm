"""Step 2: the 17 lesion findings on one CT, one call per item.

Answer keys: vista_bench per-CT findings read from the scan's own report. Each findings question
carries the fixed 'how the answer key was made' block (category definitions from the
report-extraction prompt, rules 8-9; vista_bench docs/radiology_findings_prompt.md). Organ = any
finding not classified benign; lobe = malignant or possibly malignant; pneumonitis from its own
extraction (Phil 10-27).
"""

from .base import ANSWER, Card, Question

KEY = ("How the answer key was made: a language model read the radiologist's report for this scan and classified each finding as "
       "malignant; possibly malignant (the report raises cancer as a real possibility, for example 'cannot exclude malignancy' or a differential that includes cancer); "
       "nonspecific (called nonspecific or indeterminate, with no lean either way); benign (reactive, inflammatory, physiologic, or a benign tumour such as a cyst or hemangioma); "
       "or not characterized. A 'non-benign finding' is any finding not classified as benign.")

PNEUMONITIS_KEY = ("How the answer key was made: a language model read the radiologist's report for this scan and marked whether it "
                   "describes non-infectious pneumonitis (radiation-, drug- or immunotherapy-related, or organizing pneumonia) seen "
                   "on this scan. Infection-type causes and past episodes mentioned only in the history do not count.")

LOBES = (("rul", "right upper lobe", "the right apex counts as the right upper lobe"),
         ("rml", "right middle lobe", ""),
         ("rll", "right lower lobe", "the right lung base counts as the right lower lobe"),
         ("lul", "left upper lobe", "the lingula and the left apex count as the left upper lobe"),
         ("lll", "left lower lobe", "the left lung base counts as the left lower lobe"))

ORGANS = (("lungs", "the lungs"), ("lymph_nodes", "the lymph nodes"), ("mediastinum", "the mediastinum"),
          ("pleura", "the pleura"), ("bone", "the bones"), ("liver", "the liver"), ("adrenal", "the adrenal glands"),
          ("kidney", "the kidneys"), ("peritoneum", "the peritoneum"), ("thyroid", "the thyroid"))


def _questions():
    qs = []
    for k, name, rule in LOBES:
        extra = f" ({rule}.)" if rule else ""
        qs.append(Question(f"lobe_{k}", f"per_ct_lung_lobe_{k}",
                           f"{KEY}\n\nQuestion: Is there a malignant or possibly malignant lung lesion in the {name}?{extra}\n{ANSWER}"))
    for k, name in ORGANS:
        qs.append(Question(f"lesion_{k}", f"per_ct_lesion_{k}",
                           f"{KEY}\n\nQuestion: Is there a non-benign finding in {name}?\n{ANSWER}"))
    qs.append(Question("primary_tumor_visible", "per_ct_primary_tumor_visible",
                       f"{KEY}\n\nQuestion: Is the patient's primary tumour visible on this scan (a finding read as the primary cancer)?\n{ANSWER}"))
    qs.append(Question("pneumonitis", "per_ct_pneumonitis",
                       f"{PNEUMONITIS_KEY}\n\nQuestion: Does this scan show non-infectious pneumonitis?\n{ANSWER}"))
    return tuple(qs)


CARD = Card(
    step=2,
    title="Lesion findings, 17 items (one call per item)",
    opening="These are axial slices of a chest CT, ordered from top (head) to bottom (feet), each shown in {windows}.",
    joiner="\n\n",
    questions=_questions(),
)
