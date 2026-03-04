# """
# Prompt templates for VLM keyword extraction (5 keywords + reasoning).
# """
# # Template placeholders: {task_query}, {patient_timeline}, {searched_keywords}
# # - patient_timeline: formatted events from previous iterations (or "No evidence retrieved yet." on first)
# # - searched_keywords: comma-separated keywords already searched (or "No previous searches." on first)
# KEYWORD_EXTRACTION_TEMPLATE = """
# You are an expert Clinical AI Research Agent. Your goal is to iteratively retrieve a patient timeline to answer a specific clinical question: {task_query}

# ### CURRENT PROGRESS
# <current_evidence>
# {patient_timeline}
# </current_evidence>

# <search_history>
# {searched_keywords}
# </search_history>

# ### INSTRUCTIONS
# 1. **Initial Step (Cold Start)**: If <current_evidence> is empty or indicates no evidence has been retrieved, focus on identifying the foundational information needed (e.g., hospital codes, initial diagnosis, the most recent radiology reports, or pathology results) to begin answering the query.
# 2. **Gap Analysis (Subsequent Turns)**: If evidence is present, analyze it against the {task_query}. Identify specific "missing links" (e.g., a specific scan date, a biopsy measurement, or a treatment start date).
# 3. **Search Strategy**: Select EXACTLY 3 search terms highly specific to the missing information or foundational data. Use clinical terminology (e.g., "RECIST" instead of "tumor size").
# 4. **Keyword Optimization**: Use medically relevant, clinical terminology (e.g., "RECIST" instead of "tumor size") to improve BM25 retrieval accuracy.
# 5. **CRITICAL - No Repeats**: The <search_history> below lists keywords ALREADY SEARCHED. You MUST output ONLY NEW keywords that are NOT in that list. Never repeat a keyword from <search_history>. If all obvious terms are already searched, try synonyms or related clinical concepts (e.g., "mortality" instead of "death", "neoplasm" instead of "cancer").

# ### RESPONSE FORMAT (STRICT)
# You must follow the format shown in the examples. Provide your reasoning inside <clinical_reasoning> tags and the final keyword list inside <answer> tags. Do not include text outside of the tags and do not include any extra text.

# ### OUTPUT EXAMPLES
# **Example 1: Initial Iteration (No Evidence)**
# <clinical_reasoning>
# No evidence has been retrieved yet. To answer whether the tumor has progressed, I first need to find the baseline imaging and the most recent CT reports to establish a comparison.
# </clinical_reasoning>
# <answer>
# ["Malignant neoplasm", "CT abdomen", "oncology notes"]
# </answer>

# **Example 2: Subsequent Iteration (Gap Analysis)**
# <clinical_reasoning>
# I will not repeat any keywords from the <search_history> tags. Current evidence shows a CT scan from 2023-01-10 but lacks a follow-up to determine progression. I need to find the next scan or any pathology reports after January 2023.
# </clinical_reasoning>
# <answer>
# ["follow-up CT", "embolism", "metformin"]
# </answer>

# **Example 3: Subsequent Iteration (Gap Analysis)**
# <clinical_reasoning>
# I will not repeat any keywords from the <search_history> tags. Currently information pertaining to hospital codes outside of radiology and pathology notes are missing.
# </clinical_reasoning>
# <answer>
# ["ICD10CM", "cancer stage", "pneumonitis"]
# </answer>
# """

# KEYWORD_EXTRACTION_TEMPLATE = """
# You are an expert Clinical AI Research Agent. Your goal is to iteratively retrieve a patient timeline to answer a specific clinical question: {task_query}

# ### INPUT CONTEXT
# <internal_state>
# {internal_state}
# </internal_state>

# <current_evidence>
# {patient_timeline}
# </current_evidence>

# <search_history>
# {searched_keywords}
# </search_history>

# ### INSTRUCTIONS
# 1. **Update Internal State**: Review the <current_evidence>. Update the <internal_state> by summarizing new confirmed findings and explicitly updating the "Missing Gaps."
# 2. **Clinical Strategy**: Based on the updated state, identify the single most critical piece of missing information.
# 3. **Keyword Generation**: Select EXACTLY 3 new search terms using clinical terminology (e.g., "RECIST", "Neoplasm").
# 4. **CRITICAL - No Repeats**: Cross-reference your selection with <search_history>. You MUST output keywords that have NOT been searched before.

# ### RESPONSE FORMAT (STRICT)
# You must follow the format shown in the examples. Do not echo the <current_evidence> or <search_history> in your response. Output only the two tags below.

# ### OUTPUT EXAMPLES

# **Example 1: Initial Iteration**
# <internal_state>
# [Confirmed]: No data retrieved yet.
# [Gaps]: Primary diagnosis, baseline imaging, and treatment history are unknown.
# </internal_state>
# <answer>
# ["Oncology consult", "Baseline CT", "Malignant neoplasm"]
# </answer>

# **Example 2: Subsequent Iteration**
# <internal_state>
# [Confirmed]: Patient diagnosed with NSCLC (2023-05-12). Nivolumab started, but specific start date is unclear. 
# [Gaps]: Missing specific treatment start date and follow-up CT from 2024 to assess progression.
# </internal_state>
# <answer>
# ["Nivolumab administration", "Radiology report 2024", "Treatment response"]
# </answer>
# """

# Retrieval try #3
KEYWORD_EXTRACTION_TEMPLATE = """
You are an expert Clinical Prognostician. You are analyzing the PRE-TREATMENT history of a patient who is diagnosed with cancer. Your goal is to retrieve clinical signals and risk factors to help answer a future-state question: 

### PREDICTIVE TASK QUESTION 
{task_query}

### RETRIEVAL ITERATION {iteration}

### CURRENT EVIDENCE (a summary of what you have retrieved so far from the patient timeline)
<current_evidence>
{patient_timeline}
</current_evidence>


### SEARCH DIARY (your clinical reasoning from previous iterations)
<search_diary>
{search_diary}
</search_diary>


### SEARCH HISTORY (already used; DO NOT repeat)
<search_history>
{searched_keywords}
</search_history>

### DECISION PROCESS (FOLLOW STRICTLY)
1) Determine retrieval phase:
   - COLD START if <current_evidence> is empty or says no evidence.
   - GAP FILL if <current_evidence> exists.
2) Review the <search_diary> to see your clinical reasoning from previous iterations. Use this to guide your gap plan and new keywords.
3) Write a 3-slot "gap plan" (one slot per keyword) in <clinical_reasoning> XML tags:
   - Slot 1: most critical missing fact to answer {task_query}
   - Slot 2: second most critical missing fact
   - Slot 3: a complementary angle (synonym, code system term, related document type, or downstream consequence)
4) Convert each slot into ONE BM25 keyword phrase:
   - Prefer concise 1–4 word phrases with specific clinical tokens.
   - Use clinical acronyms/codes when helpful (RECIST, ICD-10, CPT, LOINC, SNOMED, etc.).
   - Include document/test types when useful (e.g., "radiology impression", "pathology report", "operative note", "discharge summary").
5) Diversity constraint: the 3 keywords must come from 3 different categories:
   A) diagnosis/staging/problem list
   B) imaging/pathology/lab/test
   C) systemic signs/procedure/outcome/toxicity
6) Novelty constraint: ALL 3 must be NEW vs <search_history>. If you are running out of new terms, use:
   - synonyms (e.g., "progression"→"RECIST progression", "metastasis"→"distant metastases")
   - alternative code terms (e.g., "TNM stage"→"AJCC stage")
   - alternative note types (e.g., "oncology note"→"tumor board note")

### RESPONSE FORMAT (STRICT)
Return ONLY the <clinical_reasoning> and <answer> tags with no extra text.

<clinical_reasoning>
Phase: [COLD START or GAP FILL]
Gap plan:
1) ...
2) ...
3) ...
</clinical_reasoning>
<answer>
["keyword 1", "keyword 2", "keyword 3"]
</answer>

### OUTPUT EXAMPLES
**Example 1: Initial Iteration (No Evidence)**
<clinical_reasoning>
Phase: COLD START
Gap plan:
1) Establish primary diagnosis and coded problem list
2) Identify baseline objective disease measurement imaging
3) Find systemic signs of disease in documentation
</clinical_reasoning>
<answer>
["ICD10 malignant neoplasm code", "baseline CT chest abdomen", "weight loss"]
</answer>

**Example 2: Subsequent Iteration (Gap Analysis)**
<clinical_reasoning>
Phase: GAP FILL
Gap plan:
1) Determine formal tumor response classification after treatment
2) Retrieve functional status assessment near treatment interval
3) Identify systemic therapy prescription
</clinical_reasoning>
<answer>
["RECIST 1.1", "ECOG status clinic note", "CRP albumin"]
</answer>

**Example 3: Subsequent Iteration (Gap Analysis)**
<clinical_reasoning>
Phase: GAP FILL
Gap plan:
1) Identify immune or treatment-related toxicity documentation
2) Retrieve laboratory monitoring associated with endocrine toxicity
3) Identify treatment intervention for toxicity management
</clinical_reasoning>
<answer>
["immune related adverse event", "TSH free T4 laboratory", "prednisone taper oncology"]
</answer>
"""

# Template for VLM summarization of patient timeline (used as current_evidence in next iteration).
# Placeholders: {task_query}, {patient_timeline}, {max_chars}
TIMELINE_SUMMARY_TEMPLATE = """
You are a clinical oncology summarization assistant. Given a patient timeline and a clinical oncology question, extract ONLY the key facts relevant to answering that question.

### CLINICAL ONCOLOGY QUESTION
{task_query}

### PATIENT TIMELINE (retrieved events)
{patient_timeline}

### INSTRUCTIONS
Summarize the timeline in a concise bullet list. Include:
- Key diagnoses, staging, and problem list items
- Important imaging findings and dates (include radiology report content when present)
- Treatment history (drugs, procedures)
- Relevant labs, biomarkers, or vital signs
- Clinical notes and note content (VALUE fields) when relevant to the question
- Any evidence towards outcomes

Give a brief summary of the timeline (under {max_chars} characters). Use clinical terminology. Do not add speculation—only facts from the timeline. Preserve key details from reports and notes.

### RESPONSE FORMAT (STRICT)
Output ONLY the summary inside <answer> tags. No other text, no thinking, no explanation outside the tags.

<answer>
[Your bullet-list summary here]
</answer>
"""