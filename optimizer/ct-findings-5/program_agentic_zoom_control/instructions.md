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

Beside the scan's ct/ folder there is organs.json: for this scan, where each organ lies along the scan's superior-inferior axis, from TotalSegmentator (an automatic organ segmenter): the first and last slice and the centre slice, and the same as fractions of the axis (0 = lowest slice, 1 = highest), which the viewer's --from/--to/--at accept. Use it to go straight to the slices you need.

LOOKING CLOSELY. Many of the findings behind these questions are small: a 5-9 mm lung nodule, lymph node or liver lesion spans only two to four slices and a few pixels on a grid of whole slices. Before you answer, look closely at each of three regions: the liver, the lungs, and the lymph nodes (the mediastinum and both hila). Do it with the viewer's --at (any slice), a small --count (fewer, larger tiles) and --crop (only a part of each slice): you must open at least one cropped view of a few slices of the liver, one of the lungs and one of the lymph nodes. Which slices, how many and where to crop are your choice. An image you drew but did not open with the Read tool was not looked at.
