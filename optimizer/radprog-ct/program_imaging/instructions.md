You are given two chest CTs of the same patient as two CT volumes: the PRIOR scan and the CURRENT scan, with the number of days between them.

Answer this question, word for word:

Question: Comparing the CURRENT scan with the PRIOR scan, has the cancer progressed?
- Yes: progression or mixed response: clear growth of known malignant lesions, or new lesions suspicious for cancer, even if other sites improved.
- No: stable disease or response.
Do not count non-cancer changes (infection, aspiration, organizing pneumonia, drug reaction, evolving post-radiation change).
Give the probability, from 0 to 100, that the answer is yes. Answer with one whole number.

To look at the scans, run `python3 -B tools/view_ct.py --help` (in this program folder) and open the images it writes
with the Read tool. It can show the PRIOR and CURRENT scans side by side, in lung, soft-tissue and bone windows.
