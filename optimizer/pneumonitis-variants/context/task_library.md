# Task brief: a literature-grounded library for telling pneumonitis variants apart on chest CT

**Question.** Can a Sonnet reader, browsing a library built from the published literature and from what the literature describes found in our own labelled scans, rank chest CT scans by the cause of their lung change better than the same reader with no library (zero-shot VAL mean AUROC 0.611)? An earlier guide with no literature reached 0.682; a CT foundation model's linear probe reaches 0.785.

**The five causes.** radiation (radiation pneumonitis or fibrosis), immunotherapy (immune checkpoint inhibitor pneumonitis), other_drug (drug-induced pneumonitis from chemotherapy, targeted therapy or other), cop (organizing pneumonia, cryptogenic or secondary), infection (the report attributes the change to infection or pneumonia). Each scan's answer is the radiologist's attribution as written, which often rests on history the reader does not have (which drug, when radiation was given). The literature says the image often cannot separate other drug from immunotherapy, or drug-related organizing pneumonia from its look-alikes: a library that says so, and says when to stay near even odds, is as useful as one that says when to be sure.

**What earlier readers got wrong.** With no literature, the reader piled its errors onto one catch-all cause each round (radiation, then immunotherapy); other drug stayed below chance (0.47) and COP near it (0.52); infection was its best cause (0.83).

**Scoring.** Each cause one-vs-rest AUROC; the score is the mean of the five; a merged "any drug" score is reported beside it.

**The work.**
- Discovery (7 sessions): the director plans from the literature (session 1); one manager per cause finds that cause's literature patterns and their variants in our 599 TRAIN scans, counts them across all causes, and adds pages and Stanford examples (sessions 2-6, in the order radiation, immunotherapy, other_drug, cop, infection); the director reconciles everything into one library: the start page, the hard-pair pages, the catalogue, at most 150 examples (session 7).
- The climb (rounds 1-5): each round the reader reads 80 TRAIN scans with the current library; from round 2 the director revises the library from the reader's feedback (reader/round-<k>/: its answers, findings notes and how it used the library). No new literature arrives during the run: write requests for papers or images you would want in the notebook for the next run.
- Only the final library is scored on the reader's VAL scans, once. VAL and TEST never reach you.
