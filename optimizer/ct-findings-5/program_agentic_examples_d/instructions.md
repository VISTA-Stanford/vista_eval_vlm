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

Beside the scan's ct/ folder there is a folder neighbours/: five patients from a separate training set whose CT scans are the most similar to this one by MerlinOnc embedding retrieval (a CT foundation model), each with its CT, its answer-key answers to these five questions and its radiology report. neighbours/README.md says what is there. Use the neighbours to learn how to read images like this one, not only as a vote: each is a worked example where you can see the scan and read what the radiologist found on it.

Three image grids have been drawn for your scan in ct_views/ and the same three for each neighbour in neighbours/n<k>/views/, all with this viewer and the same settings: lungs.png (axial, lung window, the bottom to the top of the lungs), liver.png (axial, liver window, the liver) and chest_soft.png (axial, soft-tissue window, the bottom of the heart to the top of the lungs: mediastinum, hila and axillae, for the lymph nodes). Each tile is labelled with its slice index; views.json beside them gives the slice range, or says the organ ranges were not available and the grid spans the whole scan.

Work back and forth between your scan and the neighbours:
1. Open your scan's three grids first. Note what you see and what you are unsure about for each question.
2. For each neighbour, open all three of its grids together with its report.txt and answers.json. Find on the images what the report describes: the nodule, the liver lesion, the enlarged node, the pneumonitis, and what it calls normal, benign or post-treatment. Work out what each looks like at this resolution and window, and what distinguishes it from things that look similar.
3. Go back to your scan and look again with those lessons, question by question. Where a comparison needs a closer look, render more views of your ct/ or of a neighbour's ct/ with tools/view_ct.py (organs.json gives the slice ranges). Repeat steps 2 and 3 as often as it helps.
4. Open all fifteen neighbour grids; a neighbour you did not look at is a missed step. In your findings note, for each of the five neighbours, write in one or two lines what its report says, what you could and could not see on its images, and the lesson you took for this scan. Then say which comparisons moved your answers.

Beside ct/ there is also organs.json: for this scan, where each organ lies along the scan's superior-inferior axis, from TotalSegmentator (an automatic organ segmenter): the first and last slice and the centre slice, and the same as fractions of the axis (0 = lowest slice, 1 = highest), which the viewer's --from/--to/--at accept. Each neighbour folder has its own organs.json, or says the ranges are not available. Use them to go straight to the slices you need.