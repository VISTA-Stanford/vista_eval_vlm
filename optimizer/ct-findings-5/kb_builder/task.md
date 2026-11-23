# Task brief: the CT knowledge base

## The tester

One sealed session per CT scan (Claude Sonnet) reads the scan and answers five questions with a probability 0-100. The answer key was made by a language model reading each scan's radiology report:

1. lesion_liver: Is there a non-benign finding in the liver?
2. lesion_lungs: Is there a non-benign finding in the lungs?
3. lesion_lymph_nodes: Is there a non-benign finding in the lymph nodes?
4. lung_lobe_rul: Is there a malignant or possibly malignant lung lesion in the right upper lobe? (The right apex counts.)
5. pneumonitis: Does this scan show non-infectious pneumonitis?

Each finding the report mentions was classified malignant; possibly malignant (cancer raised as a real possibility); nonspecific (indeterminate, no lean); benign (reactive, inflammatory, physiologic, or a benign tumour such as a cyst or hemangioma); or not characterized. A "non-benign finding" is any finding not classified benign. Pneumonitis means the report describes non-infectious pneumonitis (radiation-, drug- or immunotherapy-related, or organizing pneumonia) seen on this scan; infection and past episodes in the history do not count.

The tester's scans: 2.5 mm axial slabs, Hounsfield units, RAS orientation. Its tools: `tools/view_ct.py` (slices as image grids) and `tools/kb.py` (your library: `list` by finding and words, sorted by how similar each example's whole scan is to the tester's own; `open` draws each view of an entry beside the same region of the tester's own scan, mapped through each scan's organ ranges, with the same window, slices and crop). Read `instructions.md` and both tools in your working folder: that is exactly what the tester sees.

The tester's scans are TRAIN scans of other patients while you build (100 a round, about half positive for lymph nodes or pneumonitis); at the end the best round's library is scored once on held-out scans.

## An entry

`library/entries/<entry>/entry.json` (the folder name: lowercase letters, digits and dashes), plus its images in the same folder:

    {"finding": "lesion_lymph_nodes",      one of the five question keys, or "general"
     "answer": "yes",                       what the answer key says for that finding on the source scan: "yes" or "no"
     "title": "Enlarged necrotic right paratracheal node",
     "words": ["paratracheal", "necrotic", "mediastinal"],   search words
     "note": "How to recognise it at this window and size, where to look, what it is easily confused with ...",
     "report_sentence": "the radiologist's words about this finding, nothing else",
     "source": "<the source scan's folder name>",
     "views": [{"organ": "heart", "plane": "axial", "window": "soft", "from": 0.6, "to": 1.0, "count": 6,
                "crop": "0.25,0.15,0.75,0.6", "image": "view1.png"}],
     "text_only": false}

A view's "organ" is one in the scan's organs.json (lungs, heart, liver, lung_upper_lobe_right, ...) or "whole". For axial views, from/to are fractions of that organ's own extent (0 = its lowest slice, 1 = its highest), so the same view lands on the same part of the organ in the tester's scan. Make each view's image with

    python3 -B tools/kb.py render <source>/<scan>/ct --organs <source>/<scan>/organs.json --view '<the view as JSON>' --out library/entries/<entry>/view1.png

and open it with Read to check it shows what the note says. An entry with no views is text-only: set "text_only": true. Use those sparingly: a rule in words is something the tester could have been told without a library; the point of this one is images.
