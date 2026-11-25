# Task brief: telling pneumonitis variants apart on chest CT

**Question.** Can a Sonnet reader, given a guide and examples that you build from the reference literature and labelled TRAIN scans, rank chest CT scans by the cause of their lung change better than the same reader with no guide and no examples?

**Scans.** Chest CTs of patients with cancer (most with lung cancer) whose radiology report attributes new or changed lung abnormality to one of five causes. Each scan's answer is one of:

| key | cause | how the answer key was made |
|---|---|---|
| radiation | radiation pneumonitis or radiation fibrosis | a language model read the report's stated cause |
| immunotherapy | immune checkpoint inhibitor pneumonitis | same |
| other_drug | drug-induced pneumonitis from another drug (chemotherapy, targeted therapy, other) | same |
| cop | organizing pneumonia (cryptogenic or secondary) | same |
| infection | the report attributes the lung change to infection or pneumonia | a model confirmed the report's wording |

The answer is the radiologist's attribution as written, which often draws on history the reader does not have (dates of radiation, which drug). The reader sees only the scan. Some causes therefore have tell-tale image signs (a radiation port), and some must be ranked from softer clues; a guide that says when to stay near even odds is as useful as one that says when to be sure.

**Scoring.** Each cause one-vs-rest AUROC over the scored scans; the score is the mean of the five. The reader's probabilities are judged by ranking, not calibration.

**The reader.** One sealed session per scan, Sonnet, no web, no history: the CT (`ct/`), `organs.json` (organ ranges), `organ_mask.npz` (lobes and liver), a grid viewer and an organ-zoom tool, your guide and your library, and `drawn_examples/` (10 side-by-side images: 2 entries per cause, picked by similarity, never from the same patient). It must open all 10 images.

**Your scans.** `scans/` holds labelled TRAIN scans, one folder each: `ct/` (ct.npz, meta.json), `organs.json`, `answer.json` (the cause), `report.txt` (the radiology report). `scans/index.json` lists each with its cause and where it came from (`source`: given for round 1; `round-<k>`: a scan the reader got wrong in round k). Look at the reports for how radiologists describe each cause; look at the images for what that looks like.

**The loop.** 5 rounds. Each round: you revise guide/ and examples/; then the reader reads 80 fresh TRAIN scans (16 per cause) with your version; its answers and findings notes come back to you, and the scans it got wrong are added to `scans/`. Only round 5's version is scored on VAL, once. VAL and TEST scans never reach you.
