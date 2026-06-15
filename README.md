---

title: Paper2Lab
emoji: 🧪
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false

tags:
  - track:backyard
  - sponsor:nvidia
  - sponsor:modal
  - achievement:offbrand
  - achievement:fieldnotes

  - backyard-ai
  - scientific-research
  - rag
  - document-ai
  - nvidia
  - reproducibility
  - gradio
  - modal
  - pdf
  - llm

---

# Paper2Lab

Turn scientific papers into structured research artifacts, reproducibility assessments, and experiment-ready lab starter kits.

## Highlights

- Tested on 40 scientific papers
- Supports Machine Learning, Clinical Research, Survey Studies, and Systematic Reviews
- Generates structured research artifacts in under 60 seconds
- Produces reproducibility assessments and experiment-ready lab starter kits
- Optional NVIDIA Nemotron refinement deployed on Modal



## Hackathon Submission

### Track

🏡 **Backyard AI**

Paper2Lab was inspired by a conversation with a biology research student who struggled to move from reading scientific papers to actually reproducing their experiments.

Researchers spend significant time extracting methodology details, identifying datasets, understanding evaluation protocols, and designing reproduction plans.

Paper2Lab automates this workflow and transforms a paper into experiment-ready research artifacts in under 60 seconds.

## Why It Matters

Researchers spend hours manually extracting datasets, methods, evaluation protocols, and reproducibility details from papers.

Paper2Lab helps researchers move from reading papers to designing experiments by automatically generating structured summaries, evidence-grounded findings, reproducibility assessments, and lab starter kits.

---

## Live Demo

**Hugging Face Space**

https://huggingface.co/spaces/build-small-hackathon/Paper2Lab

---

## Demo Video

https://drive.google.com/file/d/1d1s7dcAjM_GdjeT4zhmqMPEH2Cxa4Sfb/view?usp=sharing

Demo includes:

- Attention Is All You Need
- Single-Cell RNA Sequencing Analysis
- NVIDIA Nemotron refinement
- Reproducibility assessment
- Lab starter kit generation

---

## Social Post

LinkedIn:

https://www.linkedin.com/feed/update/urn:li:ugcPost:7472403996360581120/

---

## GitHub repository

GitHub:

https://github.com/miranitta/Paper2Lab

---

## Screenshots

### Landing Page

![Landing Page](assets/landing.png)

### Quick Summary

![Quick Summary](assets/quick-summary.png)

### Summary

![Summary](assets/summary.png)

### Lab Starter Kit

![Lab Starter Kit](assets/lab-kit.png)

### Evidence Viewer

![Evidence Viewer](assets/evidence.png)

### Ask The Paper

![Ask The Paper](assets/ask-paper.png)

### Advanced Analysis

![Advanced Analysis](assets/advanced.png)

### Export

![Export](assets/export.png)

### Modal deployment

![Modal deployment](assets/modal-deploy.png)

---

## Team

Solo Submission

Hugging Face Username: RLazreg

---

## What Paper2Lab Generates

Upload a scientific paper and automatically obtain:

* Structured Paper Card
* Evidence-Grounded Summary
* Dataset Extraction
* Model & Method Extraction
* Reproducibility Assessment
* Experiment Roadmap
* Lab Starter Kit
* Interactive Question Answering
* Exportable JSON Reports
* Exportable Markdown Reports

---

## Key Features

### Structured Paper Understanding

Automatically extracts:

* Research Question
* Contributions
* Methodology
* Datasets
* Models and Methods
* Metrics
* Findings
* Limitations

### Evidence Grounding

Every extraction is linked to supporting evidence retrieved directly from the paper.

### Ask the Paper

Ask questions such as:

* What dataset was used?
* What model was proposed?
* What metrics were reported?
* What limitations were identified?

### Reproducibility Assessment

Evaluates:

* Dataset availability
* Experimental setup quality
* Hyperparameter reporting
* Evaluation completeness
* Code availability

### Lab Starter Kit

Generates:

* Project structure
* Required dependencies
* Dataset plan
* Experiment checklist
* Evaluation plan
* Reproducibility risks

---

## Technology Stack

* Python
* Gradio
* PyMuPDF
* Sentence Transformers
* Local Semantic Search
* NVIDIA Nemotron
* Modal
* Hugging Face

---

## Evaluation

Paper2Lab was tested on **40 scientific papers** spanning:

* Machine Learning
* Clinical Research
* Survey Studies
* Systematic Reviews
* General Scientific Research

Results:

* End-to-end analysis in under 60 seconds
* Structured information extraction
* Reproducibility assessment
* Experiment-ready lab starter kits
* Evidence-grounded responses

---

## Architecture

PDF
→ PyMuPDF Extraction
→ Evidence Indexing
→ Structured Paper Card
→ Reproducibility Assessment
→ Lab Starter Kit

By choice:
→ NVIDIA Nemotron Refinement (via Modal)

---

## Project Structure

paper2lab/
├── app.py
├── requirements.txt
├── modal_refine.py
├── src/
│   └── paper2lab/
│       ├── data/
│       │   └── pdf_loader.py
│       ├── inference/
│       │   ├── pipeline.py
│       │   ├── paper_card.py
│       │   ├── roadmap.py
│       │   ├── lab_starter_kit.py
│       │   ├── refinement.py
│       │   └── nemotron_refiner.py
│       ├── rag/
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   └── qa.py
│       └── evaluation/
│           └── reproducibility.py
└── README.md

---

## How It Works

1. Upload a PDF paper
2. Extract paper content
3. Build evidence index
4. Generate structured paper card
5. Optional NVIDIA Nemotron refinement
6. Run reproducibility analysis
7. Generate lab starter kit
8. Export results

---

## Installation

git clone https://github.com/miranitta/Paper2Lab.git
cd Paper2Lab

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python app.py

---

## Future Work

* Multi-paper comparison
* Citation graph exploration
* Agentic research workflows
* Multi-document RAG
* Fine-tuned extraction models

---

Built for researchers, students, engineers, and scientific teams who want to move from reading papers to running experiments.
