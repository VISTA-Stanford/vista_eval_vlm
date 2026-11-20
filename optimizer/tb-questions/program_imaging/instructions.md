You are given this patient's most recent chest CT before their thoracic tumor board and the prior CT, as two CT volumes: PRIOR and MOST RECENT.

Answer this question, word for word:

Question: As of one week before the tumor board, does the most recent imaging show cancer progression?
- Yes: progression.
- No: response or stable disease.
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

To look at the scans, run `python3 -B tools/view_ct.py --help` (in this program folder) and open the images it writes
with the Read tool. It can show the PRIOR and MOST RECENT scans side by side, in lung, soft-tissue and bone windows.
