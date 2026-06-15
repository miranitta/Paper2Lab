from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List

import requests


NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


LIST_FIELDS = [
    "contributions",
    "methodology",
    "datasets_or_data_sources",
    "models_or_methods",
    "metrics_or_measurements",
    "key_findings",
    "limitations",
    "missing_reproducibility_info",
]


def _clean_string(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:\n\t")


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json", "", text).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Nemotron returned invalid JSON.")
        return json.loads(match.group(0))


def _clean_list(items: Any, max_items: int = 10) -> List[str]:
    if not isinstance(items, list):
        return []

    out: List[str] = []
    seen = set()

    for item in items:
        if isinstance(item, dict):
            item = (
                item.get("value")
                or item.get("description")
                or item.get("text")
                or item.get("summary")
                or ""
            )

        text = _clean_string(item)
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

        if not text or key in seen:
            continue

        seen.add(key)
        out.append(text)

        if len(out) >= max_items:
            break

    return out


def _filter_datasets(items: List[str]) -> List[str]:
    canonical_sources = {
        "pubmed": "PubMed",
        "scopus": "Scopus",
        "web of knowledge": "Web of Knowledge",
        "web of science": "Web of Science",
        "google scholar": "Google Scholar",
        "cochrane": "Cochrane",
        "cochrane library": "Cochrane Library",
        "embase": "Embase",
        "medline": "MEDLINE",
        "eric": "ERIC",
        "clinicaltrials": "ClinicalTrials.gov",
    }

    hard_reject = [
        "limitation", "limitations", "ecological design", "classification",
        "spatial", "temporal", "errors", "overfitting", "pseudo-accuracy",
        "beam size", "during inference", "dropout", "optimizer",
        "learning rate", "attention key size",
    ]

    model_only = {
        "rnn", "lstm", "gru", "transformer", "parser",
        "berkeleyparser", "berkleyparser", "baseline",
        "architecture", "model",
    }

    found: List[str] = []

    for item in items:
        clean = _clean_string(item)
        low = clean.lower()

        if not clean:
            continue

        if any(bad in low for bad in hard_reject):
            continue

        if low in model_only:
            continue

        # Canonical exact/word-boundary source extraction.
        for key, label in canonical_sources.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", low):
                found.append(label)

        # Keep short explicit dataset/source phrases only.
        if len(clean.split()) <= 8 and re.search(
            r"\b(dataset|corpus|benchmark|database|registry|repository|cohort|records|patients|participants)\b",
            low,
        ):
            found.append(clean)

    return list(dict.fromkeys(found))


def _build_lab_starter_kit(card: Dict[str, Any]) -> Dict[str, Any]:
    paper_type = (card.get("paper_type") or "general_research").lower()

    datasets = card.get("datasets_or_data_sources") or []
    methods = card.get("models_or_methods") or []
    methodology = card.get("methodology") or []
    metrics = card.get("metrics_or_measurements") or []
    missing = card.get("missing_reproducibility_info") or []
    roadmap = card.get("reproduction_roadmap") or {}

    # Prefer roadmap datasets when card datasets are empty.
    roadmap_datasets = roadmap.get("datasets") if isinstance(roadmap, dict) else []
    if not datasets and isinstance(roadmap_datasets, list):
        datasets = roadmap_datasets

    blob = " ".join(
        str(x) for x in (datasets + methods + methodology + metrics)
    ).lower()

    base_project_structure = [
        "paper2lab_project/",
        "paper2lab_project/data/",
        "paper2lab_project/configs/",
        "paper2lab_project/src/",
        "paper2lab_project/outputs/",
        "paper2lab_project/README.md",
    ]

    requirements = [
        "python>=3.10",
        "numpy",
        "pandas",
        "matplotlib",
    ]

    if paper_type == "machine_learning":
        ml_requirements = requirements + [
            "scikit-learn",
        ]

        if any(x in blob for x in ["transformer", "attention", "bert", "gpt", "neural", "pytorch"]):
            ml_requirements += [
                "torch",
                "transformers",
                "datasets",
                "tokenizers",
                "evaluate",
            ]

        if "tensorflow" in blob or "keras" in blob:
            ml_requirements.append("tensorflow")

        if any(x in blob for x in ["bleu", "translation", "wmt"]):
            ml_requirements += ["sacrebleu", "sentencepiece"]

        ml_requirements = list(dict.fromkeys(ml_requirements))

        hyperparams = [
            x for x in methodology
            if any(k in x.lower() for k in [
                "learning rate", "batch", "epoch", "optimizer",
                "dropout", "warmup", "steps", "gpu", "label smoothing"
            ])
        ]

        return {
            "starter_type": "machine_learning",
            "project_structure": base_project_structure + [
                "paper2lab_project/src/preprocess.py",
                "paper2lab_project/src/train.py",
                "paper2lab_project/src/evaluate.py",
                "paper2lab_project/configs/train_config.yaml",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": ml_requirements,
            "dataset_plan": datasets or ["Dataset/source not clearly specified."],
            "training_configuration": {
                "model_or_method": methods[:6] or ["Model/method not clearly specified."],
                "hyperparameters": hyperparams or [
                    "Hyperparameters are incomplete or not clearly specified."
                ],
            },
            "experiment_checklist": [
                "Download or prepare the reported datasets.",
                "Reproduce preprocessing/tokenization steps.",
                "Implement the reported model or method.",
                "Configure training hyperparameters.",
                "Run training or analysis pipeline.",
                "Evaluate using the reported metrics.",
                "Compare reproduced outputs with paper results.",
                "Document missing details and deviations.",
            ],
            "evaluation_plan": metrics or ["Evaluation metrics not clearly specified."],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    if paper_type == "systematic_review":
        review_requirements = list(dict.fromkeys(requirements + [
            "openpyxl",
            "python-docx",
        ]))

        return {
            "starter_type": "systematic_review",
            "project_structure": base_project_structure + [
                "paper2lab_project/data/search_results/",
                "paper2lab_project/data/screening/",
                "paper2lab_project/src/search_strategy.py",
                "paper2lab_project/src/deduplicate.py",
                "paper2lab_project/src/screening_table.py",
                "paper2lab_project/src/quality_assessment.py",
                "paper2lab_project/outputs/prisma_flow.md",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": review_requirements,
            "search_strategy": datasets or ["Bibliographic databases not clearly specified."],
            "screening_checklist": [
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
            "inclusion_exclusion_criteria": [
                x for x in methodology
                if any(k in x.lower() for k in [
                    "inclusion", "exclusion", "eligibility", "criteria"
                ])
            ] or ["Inclusion/exclusion criteria not clearly specified."],
            "quality_assessment_tools": methods or [
                "Quality assessment tool not clearly specified."
            ],
            "evaluation_plan": metrics or [
                "Number of records identified.",
                "Number of included studies.",
                "Quality assessment summary.",
            ],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    if paper_type == "clinical_study":
        clinical_requirements = list(dict.fromkeys(requirements + [
            "scipy",
            "statsmodels",
            "openpyxl",
        ]))

        return {
            "starter_type": "clinical_study",
            "project_structure": base_project_structure + [
                "paper2lab_project/data/raw/",
                "paper2lab_project/data/processed/",
                "paper2lab_project/src/cohort_selection.py",
                "paper2lab_project/src/statistical_analysis.py",
                "paper2lab_project/src/outcome_analysis.py",
                "paper2lab_project/outputs/tables/",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": clinical_requirements,
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
            "evaluation_plan": metrics or [
                "Outcome measurement plan not clearly specified."
            ],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    if paper_type in {"survey_paper", "review_paper", "survey_study", "guide_or_report"}:
        survey_requirements = list(dict.fromkeys(requirements + [
            "openpyxl",
            "python-docx",
        ]))

        return {
            "starter_type": "survey_or_review",
            "project_structure": base_project_structure + [
                "paper2lab_project/data/literature/",
                "paper2lab_project/src/literature_mapping.py",
                "paper2lab_project/src/comparison_matrix.py",
                "paper2lab_project/src/synthesis_report.py",
                "paper2lab_project/outputs/comparison_matrix.xlsx",
                "paper2lab_project/requirements.txt",
            ],
            "requirements_txt": survey_requirements,
            "literature_mapping_plan": datasets or [
                "Literature sources not clearly specified."
            ],
            "survey_dimensions": methodology or [
                "Survey/review dimensions not clearly specified."
            ],
            "comparison_framework": methods or [
                "Comparison framework not clearly specified."
            ],
            "evaluation_plan": metrics or [
                "Synthesis/evaluation criteria not clearly specified."
            ],
            "reproducibility_risks": missing or ["No major missing information detected."],
        }

    return {
        "starter_type": "general_research",
        "project_structure": base_project_structure + [
            "paper2lab_project/src/reproduce.py",
            "paper2lab_project/src/evaluate.py",
            "paper2lab_project/requirements.txt",
        ],
        "requirements_txt": list(dict.fromkeys(requirements)),
        "dataset_plan": datasets or ["Dataset/source not clearly specified."],
        "method_or_procedure": methodology or methods or [
            "Method/procedure not clearly specified."
        ],
        "evaluation_plan": metrics or ["Evaluation metrics not clearly specified."],
        "reproducibility_risks": missing or ["No major missing information detected."],
    }


def _compact_evidence_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    candidate = copy.deepcopy(pack.get("candidate_paper_card", {}))

    compact_sections = []
    for sec in pack.get("section_previews", [])[:12]:
        compact_sections.append({
            "title": sec.get("title"),
            "role_hint": sec.get("role_hint"),
            "page_start": sec.get("page_start"),
            "page_end": sec.get("page_end"),
            "preview": _clean_string(sec.get("preview", ""))[:1800],
        })

    return {
        "candidate_paper_card": candidate,
        "section_previews": compact_sections,
        "captions": pack.get("captions", [])[:8],
        "tables": pack.get("tables", [])[:3],
        "metadata": pack.get("metadata", {}),
    }


def validate_refined_card(refined: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    final = copy.deepcopy(fallback)

    for key, value in refined.items():
        if key == "llm_evidence_pack":
            continue
        final[key] = value

    for field in LIST_FIELDS:
        final[field] = _clean_list(final.get(field))

    final["datasets_or_data_sources"] = _filter_datasets(
        final.get("datasets_or_data_sources", [])
    )

    final["title"] = _clean_string(final.get("title")) or fallback.get("title")
    final["field"] = _clean_string(final.get("field")) or fallback.get("field")
    final["paper_type"] = _clean_string(final.get("paper_type")) or fallback.get("paper_type")
    final["research_question"] = (
        _clean_string(final.get("research_question"))
        or fallback.get("research_question")
    )

    final["annotation_version"] = fallback.get("annotation_version", "v1.0")
    final["source_pdf"] = fallback.get("source_pdf")
    final["metadata"] = fallback.get("metadata", {})

    if not isinstance(final.get("lab_starter_kit"), dict):
        final["lab_starter_kit"] = _build_lab_starter_kit(final)
    if not isinstance(final.get("lab_starter_kit"), dict):
        final["lab_starter_kit"] = _build_lab_starter_kit(final)
    return final


def diff_cards(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    changed, added, removed = [], [], []

    for key in sorted(set(before.keys()) | set(after.keys())):
        if key == "llm_evidence_pack":
            continue
        if key not in before:
            added.append(key)
        elif key not in after:
            removed.append(key)
        elif before.get(key) != after.get(key):
            changed.append(key)

    return {
        "changed_fields": changed,
        "added_fields": added,
        "removed_fields": removed,
    }


def _call_nvidia(prompt: str, model: str, timeout: int = 180) -> str:
    modal_url = os.getenv("MODAL_REFINE_URL")
    print("MODAL_REFINE_URL:", modal_url)

    # =========================
    # Use Modal if configured
    # =========================
    if modal_url:
        response = requests.post(
            modal_url,
            json={
                "prompt": prompt,
                "model": model,
            },
            timeout=timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Modal error {response.status_code}: {response.text[:1000]}"
            )

        data = response.json()

        if data.get("status") != "ok":
            raise RuntimeError(
                f"Modal refinement failed: {data}"
            )

        return data["content"]

    # =========================
    # Fallback: Direct NVIDIA
    # =========================
    api_key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NVIDIA_API_KEY".lower())
        or os.getenv("nvidia-api-key")
    )

    if not api_key:
        raise RuntimeError(
            "Missing NVIDIA_API_KEY. Set it in your environment before using refinement_mode='nemotron'."
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise scientific JSON refiner. "
                    "Return only valid JSON. No markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "top_p": 0.7,
        "max_tokens": 8192,
    }

    response = requests.post(
        NVIDIA_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"NVIDIA API error {response.status_code}: {response.text[:1000]}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected NVIDIA response: {data}"
        ) from exc


def refine_with_nemotron(
    llm_evidence_pack: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    return_comparison: bool = True,
) -> Dict[str, Any]:
    if "candidate_paper_card" not in llm_evidence_pack:
        raise ValueError("llm_evidence_pack must contain candidate_paper_card.")

    compact_pack = _compact_evidence_pack(llm_evidence_pack)
    before = copy.deepcopy(compact_pack["candidate_paper_card"])

    prompt = f"""
You are Paper2Lab.

Refine candidate_paper_card using ONLY the provided evidence.

Return ONLY valid JSON.

Strict rules:
- Do not invent facts.
- Do not add facts that are not in the evidence pack.
- Remove boilerplate, references, author contributions, affiliations, acknowledgements, and duplicate claims.
- datasets_or_data_sources must contain only real datasets, corpora, benchmarks, databases, or data sources.
- Do NOT put models, parsers, methods, architectures, baselines, algorithms, or metrics in datasets_or_data_sources.
- Preserve annotation_version.
- Preserve source_pdf.
- Keep outputs concise.
- If evidence is insufficient, use [] or null.
- Add or improve lab_starter_kit.

Return a compact JSON object with ONLY these keys:
- title
- field
- paper_type
- research_question
- contributions
- methodology
- datasets_or_data_sources
- models_or_methods
- metrics_or_measurements
- key_findings
- limitations
- missing_reproducibility_info
- reproduction_roadmap
- reproducibility_score
- lab_starter_kit
- source_pdf
- annotation_version

Do not return metadata.
Do not return long nested evidence_terms.
Do not repeat large diagnostics objects.
Keep every list to maximum 6 items.

Evidence pack:
{json.dumps(compact_pack, indent=2, ensure_ascii=False)}
""".strip()

    raw = _call_nvidia(prompt=prompt, model=model)

    try:
        refined_raw = _extract_json(raw)
    except Exception as exc:
        return {
            "status": "error",
            "model": model,
            "error": str(exc),
            "before_refinement": before,
            "raw_model_output": raw,
        }

    after = validate_refined_card(refined_raw, before)

    if not return_comparison:
        return after

    return {
        "status": "ok",
        "model": model,
        "before_refinement": before,
        "after_refinement": after,
        "diff_summary": diff_cards(before, after),
    }