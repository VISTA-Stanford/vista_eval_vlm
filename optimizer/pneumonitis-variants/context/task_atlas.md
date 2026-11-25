# Task brief: an atlas for telling pneumonitis variants apart on chest CT

**Question.** Can a Sonnet reader, given a literature review, a guide and a fixed atlas of clear, marked examples of each cause that you build from the literature and labelled TRAIN scans, rank chest CT scans by the cause of their lung change better than the same reader with no guide and no atlas?

**Scans.** Chest CTs of patients with cancer (most with lung cancer) whose radiology report attributes new or changed lung abnormality to one of five causes. Each scan's answer is one of:

| key | cause | how the answer key was made |
|---|---|---|
| radiation | radiation pneumonitis or radiation fibrosis | a language model read the report's stated cause |
| immunotherapy | immune checkpoint inhibitor pneumonitis | same |
| other_drug | drug-induced pneumonitis from another drug (chemotherapy, targeted therapy, other) | same |
| cop | organizing pneumonia (cryptogenic or secondary) | same |
| infection | the report attributes the lung change to infection or pneumonia | a model confirmed the report's wording |

The answer is the radiologist's attribution as written, which often draws on history the reader does not have (dates of radiation, which drug). The reader sees only the scan. Some causes have tell-tale image signs (a radiation port); some must be ranked from softer clues; a guide that says when to stay near even odds is as useful as one that says when to be sure.

**Scoring.** Each cause one-vs-rest AUROC over the scored scans; the score is the mean of the five. Judged by ranking, not calibration.

**The reader.** One sealed session per scan, Sonnet, no web, no history: the CT (`ct/`), `organs.json`, `organ_mask.npz`, a grid viewer and an organ-zoom tool, your guide and literature review, and `atlas_images/`: all 50 atlas images with an index of each entry's findings and slices (never an entry from its own patient). It must open every atlas image and, for each cause, check its scan for that cause's patterns where the guide says they appear.

**Your scans.** `scans/` holds labelled TRAIN scans, one folder each: `ct/` (ct.npz, meta.json), `organs.json`, `answer.json` (the cause), `report.txt` (the radiology report). `scans/index.json` lists each with its cause and where it came from (`source`: given before round 1; `round-<k>`: a scan the reader got wrong in round k).

**The literature.** guide/literature.md arrives with a literature review (52 sources from the published literature, found with web search outside this loop). You cannot search the web here.

**The loop.** Round 0: a first guide built from the literature review (no atlas yet; the reader does not run). Rounds 1 to 5: you build or revise the atlas, the guide and the review; then the reader reads 80 fresh TRAIN scans (16 per cause) with your version; its answers and findings notes come back, and the scans it got wrong are added to `scans/`. Only round 5's version is scored on VAL, once. VAL and TEST scans never reach you.
