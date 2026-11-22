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

Beside the scan's ct/ folder there is a folder neighbours/: five patients from a separate training set whose CT scans are the most similar to this one by MerlinOnc embedding retrieval (a CT foundation model), each with its CT, its answer-key answers to these five questions and its radiology report. neighbours/README.md says what is there. Before you answer, view at least one slice of each of the five neighbours' CTs with the same viewer (their ct/ folders are in the same format as your scan). Beyond that, use them however helps.

Beside ct/ there is also organs.json: for this scan, where each organ lies along the scan's superior-inferior axis, from TotalSegmentator (an automatic organ segmenter): the first and last slice and the centre slice, and the same as fractions of the axis (0 = lowest slice, 1 = highest), which the viewer's --from/--to/--at accept. Each neighbour folder has its own organs.json, or says the ranges are not available. Use them to go straight to the slices you need.