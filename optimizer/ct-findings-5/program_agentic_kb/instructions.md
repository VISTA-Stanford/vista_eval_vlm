You are a radiologist reading one CT scan of a patient with cancer.

How the answer key was made: a language model read the radiologist's report for this scan and classified each finding as malignant; possibly malignant (the report raises cancer as a real possibility, for example 'cannot exclude malignancy' or a differential that includes cancer); nonspecific (called nonspecific or indeterminate, with no lean either way); benign (reactive, inflammatory, physiologic, or a benign tumour such as a cyst or hemangioma); or not characterized. A 'non-benign finding' is any finding not classified as benign. For pneumonitis, it marked whether the report describes non-infectious pneumonitis (radiation-, drug- or immunotherapy-related, or organizing pneumonia) seen on this scan; infection-type causes and past episodes mentioned only in the history do not count.

QUESTIONS:
1. lesion_liver: Is there a non-benign finding in the liver?
2. lesion_lungs: Is there a non-benign finding in the lungs?
3. lesion_lymph_nodes: Is there a non-benign finding in the lymph nodes?
4. lung_lobe_rul: Is there a malignant or possibly malignant lung lesion in the right upper lobe? (The right apex counts as the right upper lobe.)
5. pneumonitis: Does this scan show non-infectious pneumonitis?

Answer each with the probability, from 0 to 100, that the answer is yes.

To look at the scan, run `python3 -B tools/view_ct.py --help` (in this program folder) and open the images it writes with the Read tool.


Beside the scan's ct/ folder there are two more files. organs.json: for this scan, where each organ lies along the scan's superior-inferior axis, from TotalSegmentator (an automatic organ segmenter): the first, last and centre slice, and the same as fractions of the axis (0 = lowest slice, 1 = highest), which the viewer's --from/--to/--at accept. kb_similarity.json: how similar this scan is, as a whole, to the scan behind each knowledge-base example (MerlinOnc, a CT foundation model).

KNOWLEDGE BASE. library/ in this program folder is a knowledge base built from other patients' scans: worked examples, each an image of a finding (or of a look-alike that is not it), the radiologist's words for it, the answer key's verdict on that scan, and a note on how to recognise it. Start with library/README.md. Then, for each question, find the examples that bear on what you see or are unsure about with `python3 -B tools/kb.py list --finding <question key> --scan <your scan's folder>` (the folder that holds ct/, organs.json and kb_similarity.json) (add `--words "..."` to narrow it), and open the useful ones with `python3 -B tools/kb.py open <entry> --scan <your scan's folder>`. Each opened example is drawn beside the same region of your own scan, with the same window and slices, so you can compare them directly: open the image, look at both halves, then go back to your own scan with tools/view_ct.py and look again where the comparison points. Work back and forth like this, example by example, before you settle each answer.

EXAMPLES DRAWN FOR YOUR SCAN. Beside ct/ there is also a folder examples/: the library's examples most similar to your scan for lymph nodes and for pneumonitis, already drawn beside the same region of your own scan (left: the example; right: your scan; same window and slices). examples/index.md lists them with each example's answer-key verdict, the radiologist's words and the note. Read examples/index.md, then open every image in examples/ with the Read tool before you answer, and compare the two halves: what does the example show, and is the same thing on your side? An example image you did not open is a missed step. Then look again at your own scan with tools/view_ct.py where the comparison points. If examples/ has no images, the library is empty: say so in your findings note and answer from your scan alone. In your findings note, for each example image, write one line: what you saw on each half and whether it changed your reading.
