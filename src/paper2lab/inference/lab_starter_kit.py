from __future__ import annotations

from typing import Any, Dict, List




KNOWN_SOURCE_NAMES = [
    "PubMed",
    "Scopus",
    "Web of Knowledge",
    "Web of Science",
    "ERIC",
    "Educational Resources and Information Center",
    "Cochrane",
    "Embase",
    "MEDLINE",
    "Google Scholar",
]


def _clean_sources(items: List[str]) -> List[str]:
    text = " ".join(str(x) for x in items).lower()
    found = []

    for name in KNOWN_SOURCE_NAMES:
        if name.lower() in text:
            found.append(name)

    if found:
        return _dedupe(found)

    # Fallback: keep only short source-like entries.
    return _dedupe([
        x for x in items
        if len(str(x).split()) <= 6
    ])

def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        item = str(item).strip()
        key = item.lower()

        if item and key not in seen:
            seen.add(key)
            out.append(item)

    return out


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value:
        return [str(value).strip()]
    return []


def _roadmap_list(roadmap: Dict[str, Any], key: str) -> List[str]:
    value = roadmap.get(key, [])

    if not isinstance(value, list):
        return _as_list(value)

    out: List[str] = []

    for item in value:
        if isinstance(item, dict):
            desc = item.get("description") or item.get("text") or item.get("step")
            if desc:
                out.append(str(desc).strip())
        elif item:
            out.append(str(item).strip())

    return _dedupe(out)


def build_lab_starter_kit(paper_card: Dict[str, Any]) -> Dict[str, Any]:
    paper_type = (paper_card.get("paper_type") or "general_research").lower()

    roadmap = paper_card.get("reproduction_roadmap", {}) or {}

    datasets = _as_list(paper_card.get("datasets_or_data_sources"))
    methods = _as_list(paper_card.get("models_or_methods"))
    methodology = _as_list(paper_card.get("methodology"))
    metrics = _as_list(paper_card.get("metrics_or_measurements"))
    missing = _as_list(paper_card.get("missing_reproducibility_info"))

    roadmap_datasets = _roadmap_list(roadmap, "datasets") if isinstance(roadmap, dict) else []
    roadmap_eval = _roadmap_list(roadmap, "evaluation_procedure") if isinstance(roadmap, dict) else []
    roadmap_steps = _roadmap_list(roadmap, "experimental_steps") if isinstance(roadmap, dict) else []

    if not datasets and roadmap_datasets:
        datasets = roadmap_datasets
    if paper_type == "systematic_review":
        datasets = _clean_sources(datasets)

    blob = " ".join(datasets + methods + methodology + metrics).lower()

    base_structure = [
        "paper2lab_project/",
        "paper2lab_project/data/",
        "paper2lab_project/configs/",
        "paper2lab_project/src/",
        "paper2lab_project/outputs/",
        "paper2lab_project/README.md",
    ]

    base_requirements = [
        "python>=3.10",
        "numpy",
        "pandas",
        "matplotlib",
    ]

    # ------------------------------------------------------------------
    # Machine-learning papers
    # ------------------------------------------------------------------
    if paper_type == "machine_learning":
        deps = base_requirements + ["scikit-learn"]

        if any(x in blob for x in ["transformer", "bert", "gpt", "neural", "attention", "pytorch"]):
            deps += ["torch", "transformers", "datasets", "tokenizers", "evaluate"]

        if any(x in blob for x in ["tensorflow", "keras"]):
            deps.append("tensorflow")

        if any(x in blob for x in ["bleu", "translation", "wmt"]):
            deps += ["sacrebleu", "sentencepiece"]

        hyperparams = [
            item for item in methodology
            if any(k in item.lower() for k in [
                "learning rate",
                "batch",
                "epoch",
                "optimizer",
                "dropout",
                "warmup",
                "steps",
                "gpu",
                "label smoothing",
            ])
        ]

        return {
            "starter_type": "machine_learning",
            "project_structure": base_structure + [
                "paper2lab_project/data/raw/",
                "paper2lab_project/data/processed/",
                "paper2lab_project/src/preprocess.py",
                "paper2lab_project/src/train.py",
                "paper2lab_project/src/evaluate.py",
                "paper2lab_project/configs/train_config.yaml",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": _dedupe(deps),
            "dataset_plan": datasets or ["Dataset/source not clearly specified."],
            "training_configuration": {
                "model_or_method": methods[:6] or ["Model/method not clearly specified."],
                "hyperparameters": hyperparams or [
                    "Hyperparameters are incomplete or not clearly specified."
                ],
            },
            "experiment_checklist": roadmap_steps or [
                "Download or prepare the reported datasets.",
                "Reproduce preprocessing/tokenization steps.",
                "Implement the reported model or method.",
                "Configure training hyperparameters.",
                "Run training or analysis pipeline.",
                "Evaluate using the reported metrics.",
                "Compare reproduced outputs with paper results.",
                "Document missing details and deviations.",
            ],
            "evaluation_plan": metrics or roadmap_eval or ["Evaluation metrics not clearly specified."],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    # ------------------------------------------------------------------
    # Systematic reviews / meta-analyses / scoping reviews
    # ------------------------------------------------------------------
    if paper_type == "systematic_review":
        deps = base_requirements + ["openpyxl", "python-docx"]

        inclusion_exclusion = [
            item for item in methodology
            if any(k in item.lower() for k in [
                "inclusion",
                "exclusion",
                "eligibility",
                "criteria",
            ])
        ]

        return {
            "starter_type": "systematic_review",
            "project_structure": base_structure + [
                "paper2lab_project/data/search_results/",
                "paper2lab_project/data/screening/",
                "paper2lab_project/src/search_strategy.py",
                "paper2lab_project/src/deduplicate.py",
                "paper2lab_project/src/screening_table.py",
                "paper2lab_project/src/quality_assessment.py",
                "paper2lab_project/outputs/prisma_flow.md",
                "paper2lab_project/outputs/synthesis_report.md",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": _dedupe(deps),
            "search_strategy": datasets or ["Bibliographic databases not clearly specified."],
            "screening_checklist": roadmap_steps or [
                "Define search query and date range.",
                "Export records from each database.",
                "Remove duplicate records.",
                "Screen titles and abstracts.",
                "Review full texts.",
                "Apply inclusion criteria.",
                "Apply exclusion criteria.",
                "Record reasons for exclusion.",
                "Build PRISMA-style flow summary.",
            ],
            "inclusion_exclusion_criteria": inclusion_exclusion or [
                "Inclusion/exclusion criteria not clearly specified."
            ],
            "quality_assessment_tools": methods or [
                "Quality assessment tool not clearly specified."
            ],
            "evaluation_plan": metrics or roadmap_eval or [
                "Number of records identified.",
                "Number of included studies.",
                "Quality assessment summary.",
            ],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    # ------------------------------------------------------------------
    # Clinical studies
    # ------------------------------------------------------------------
    if paper_type == "clinical_study":
        deps = base_requirements + ["scipy", "statsmodels", "openpyxl"]

        return {
            "starter_type": "clinical_study",
            "project_structure": base_structure + [
                "paper2lab_project/data/raw/",
                "paper2lab_project/data/processed/",
                "paper2lab_project/src/cohort_selection.py",
                "paper2lab_project/src/statistical_analysis.py",
                "paper2lab_project/src/outcome_analysis.py",
                "paper2lab_project/outputs/tables/",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": _dedupe(deps),
            "cohort_design": {
                "population_or_data_source": datasets or [
                    "Cohort/data source not clearly specified."
                ],
                "outcomes": metrics or [
                    "Clinical outcomes/endpoints not clearly specified."
                ],
            },
            "data_collection_plan": methodology or [
                "Data collection procedure not clearly specified."
            ],
            "analysis_plan": methods or [
                "Statistical analysis method not clearly specified."
            ],
            "evaluation_plan": metrics or roadmap_eval or [
                "Outcome measurement plan not clearly specified."
            ],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    # ------------------------------------------------------------------
    # Surveys, narrative reviews, guides, reports
    # ------------------------------------------------------------------
    if paper_type in {"survey_paper", "review_paper", "survey_study", "guide_or_report", "survey_or_review"}:
        deps = base_requirements + ["openpyxl", "python-docx"]

        return {
            "starter_type": "survey_or_review",
            "project_structure": base_structure + [
                "paper2lab_project/data/literature/",
                "paper2lab_project/src/literature_mapping.py",
                "paper2lab_project/src/comparison_matrix.py",
                "paper2lab_project/src/synthesis_report.py",
                "paper2lab_project/outputs/comparison_matrix.xlsx",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": _dedupe(deps),
            "literature_mapping_plan": datasets or [
                "Literature sources not clearly specified."
            ],
            "survey_dimensions": methodology or [
                "Survey/review dimensions not clearly specified."
            ],
            "comparison_framework": methods or [
                "Comparison framework not clearly specified."
            ],
            "evaluation_plan": metrics or roadmap_eval or [
                "Synthesis/evaluation criteria not clearly specified."
            ],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------
    return {
        "starter_type": "general_research",
        "project_structure": base_structure + [
            "paper2lab_project/src/reproduce.py",
            "paper2lab_project/src/evaluate.py",
            "paper2lab_project/requirements.txt",
        ],
        "requirements_txt": _dedupe(base_requirements),
        "dataset_plan": datasets or ["Dataset/source not clearly specified."],
        "method_or_procedure": methodology or methods or [
            "Method/procedure not clearly specified."
        ],
        "evaluation_plan": metrics or roadmap_eval or ["Evaluation metrics not clearly specified."],
        "reproducibility_risks": missing or ["No major missing information detected."],
    }