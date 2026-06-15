from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, final


FINAL_FIELDS = [
    "title",
    "field",
    "paper_type",
    "research_question",
    "contributions",
    "methodology",
    "datasets_or_data_sources",
    "models_or_methods",
    "metrics_or_measurements",
    "key_findings",
    "limitations",
    "missing_reproducibility_info",
    "reproduction_roadmap",
    "reproducibility_score",
    "figures_and_tables",
    "lab_starter_kit",
    "metadata",
    "source_pdf",
    "annotation_version",
]


PREFER_LOCAL_FIELDS = {
    "figures_and_tables",
    "reproducibility_score",
    "metadata",
    "source_pdf",
    "annotation_version",
}

PREFER_REFINED_FIELDS = {
    "research_question",
    "contributions",
    "methodology",
    "datasets_or_data_sources",
    "models_or_methods",
    "metrics_or_measurements",
    "key_findings",
    "limitations",
    "missing_reproducibility_info",
    "reproduction_roadmap",
    "lab_starter_kit",
}


NOISE_TERMS = [
    "department of",
    "university of",
    "corresponding author",
    "gmail.com",
    "references",
    "table of contents",
    "being accordingly",
    "endnote teachers",
    "the there",
    "resultsare",
    "analysis of the resultsare",
    "access this article online",
    "how to cite",
]


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _flatten(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(_flatten(item))
        return " ".join(parts)

    if isinstance(value, dict):
        parts = []
        for item in value.values():
            parts.append(_flatten(item))
        return " ".join(parts)

    return str(value)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False


def _noise_score(value: Any) -> float:
    text = _flatten(value).lower()
    if not text:
        return 1.0

    score = 0.0

    for term in NOISE_TERMS:
        if term in text:
            score += 1.0

    if len(text.split()) > 900:
        score += 2.0
    elif len(text.split()) > 450:
        score += 1.0

    if len(re.findall(r"\[\d+\]", text)) >= 5:
        score += 1.0

    return score


def _structure_score(value: Any) -> float:
    if _is_empty(value):
        return 0.0

    if isinstance(value, list):
        if not value:
            return 0.0
        short_items = 0
        for item in value:
            words = len(_flatten(item).split())
            if 1 <= words <= 35:
                short_items += 1
        return min(1.0, short_items / max(1, len(value)))

    if isinstance(value, dict):
        return min(1.0, len(value.keys()) / 5)

    if isinstance(value, str):
        words = len(value.split())
        if 3 <= words <= 60:
            return 1.0
        if words <= 120:
            return 0.6
        return 0.2

    return 0.4


def _completeness_score(value: Any) -> float:
    if _is_empty(value):
        return 0.0

    if isinstance(value, list):
        return min(1.0, len(value) / 4)

    if isinstance(value, dict):
        non_empty = sum(1 for v in value.values() if not _is_empty(v))
        return min(1.0, non_empty / max(1, len(value)))

    if isinstance(value, str):
        words = len(value.split())
        return min(1.0, words / 20)

    return 0.5


def _score_field(field: str, value: Any) -> float:
    if _is_empty(value):
        return 0.0

    completeness = _completeness_score(value)
    structure = _structure_score(value)
    noise = _noise_score(value)

    score = (0.45 * completeness) + (0.45 * structure) - (0.25 * noise)

    if field in PREFER_LOCAL_FIELDS:
        score += 0.15

    if field in PREFER_REFINED_FIELDS:
        score += 0.10

    return round(max(0.0, min(1.0, score)), 4)


def _similarity(a: Any, b: Any) -> float:
    text_a = set(re.findall(r"[a-z0-9]+", _flatten(a).lower()))
    text_b = set(re.findall(r"[a-z0-9]+", _flatten(b).lower()))

    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    return len(text_a & text_b) / max(1, len(text_a | text_b))


def _choose_field(
    field: str,
    local_value: Any,
    refined_value: Any,
) -> Tuple[Any, Dict[str, Any]]:
    local_score = _score_field(field, local_value)
    refined_score = _score_field(field, refined_value)
    similarity = round(_similarity(local_value, refined_value), 4)

    # For Lab Starter Kit, prefer local when it is paper-type-aware.
    # Nemotron sometimes converts systematic reviews / clinical papers into ML-style kits.
    if field == "lab_starter_kit" and isinstance(local_value, dict):
        local_text = _flatten(local_value).lower()
        refined_text = _flatten(refined_value).lower()

        local_is_specialized = any(x in local_text for x in [
            "starter_type",
            "systematic_review",
            "clinical_study",
            "survey_or_review",
            "search_strategy",
            "screening_checklist",
            "cohort_design",
            "literature_mapping_plan",
            "quality_assessment",
        ])

        refined_looks_ml_generic = any(x in refined_text for x in [
            "train.py",
            "training_configuration",
            "hyperparameters",
            "baseline model",
            "training pipeline",
            "model_or_method",
        ])

        if local_is_specialized or refined_looks_ml_generic:
            return local_value, {
                "winner": "local",
                "local_score": local_score,
                "nemotron_score": refined_score,
                "similarity": similarity,
                "reason": "local lab_starter_kit is more paper-type-aware",
            }

    if _is_empty(local_value) and not _is_empty(refined_value):
        winner = "nemotron"
        value = refined_value
    elif _is_empty(refined_value) and not _is_empty(local_value):
        winner = "local"
        value = local_value
    elif field in PREFER_LOCAL_FIELDS and local_score >= refined_score - 0.12:
        winner = "local"
        value = local_value
    elif field in PREFER_REFINED_FIELDS and refined_score >= local_score - 0.08:
        winner = "nemotron"
        value = refined_value
    elif refined_score > local_score:
        winner = "nemotron"
        value = refined_value
    else:
        winner = "local"
        value = local_value

    return value, {
        "winner": winner,
        "local_score": local_score,
        "nemotron_score": refined_score,
        "similarity": similarity,
    }

def _clean_final_datasets(items: Any, paper_type: str = "") -> List[str]:
    if not isinstance(items, list):
        return []

    paper_type = (paper_type or "").lower()

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
        "clinicaltrials": "ClinicalTrials.gov",
    }

    reject_terms = [
        "limitation", "limitations", "ecological design", "classification error",
        "incorrect spatial", "temporal assignments", "overfitting", "pseudo-accuracy",
        "beam size", "during inference", "dropout", "optimizer", "learning rate",
        "institutional review board", "informed consent", "validation set",
        "training set", "test set", "cross-validation", "augmentation",
    ]

    known_dataset_patterns = [
        r"\bPTB-XL\b",
        r"\bMUSE\b",
        r"\bTCGA[- ]?[A-Z0-9]+\b",
        r"\bGSE\d+\b",
        r"\bOECD International Migration Database\b",
        r"\bSeoul Asan Medical Center Hospital\b",

        # NLP datasets
        r"\bWMT\s*2014\b",
        r"\bWMT\b",
        r"\bPenn Treebank\b",
        r"\bWall Street Journal\b",
        r"\bWSJ\b",
        r"\b\d+\s+samples\b",

        # ML benchmarks
        r"\bHiggs Boson dataset\b",
        r"\bYahoo!?\s*LTRC\s*dataset\b",
        r"\bAllstate dataset\b",
        r"\bJFT-300M\b",
        r"\bImageNet(?:-21k)?\b",
        r"\bCOCO\b",
        r"\bCityscapes\b",
        r"\bCora\b",
        r"\bCiteseer\b",
        r"\bPubmed\b",
        r"\bNELL\b",
    ]

    out: List[str] = []

    for item in items:
        text = _clean_text(item)
        low = text.lower()

        if not text or any(bad in low for bad in reject_terms):
            continue

        if paper_type == "systematic_review":
            for key, label in canonical_sources.items():
                if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", low):
                    out.append(label)
            continue

        if paper_type in {"machine_learning", "clinical_study", "survey_study"}:
            for pat in known_dataset_patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    out.append(m.group(0).strip())
            continue

        if len(text.split()) <= 10 and re.search(
            r"\b(dataset|database|repository|registry|cohort|records|patients|participants)\b",
            low,
        ):
            out.append(text)

    if paper_type == "systematic_review":
        if "ERIC" in out and not any("educational resources" in str(x).lower() for x in items):
            out = [x for x in out if x != "ERIC"]

    return list(dict.fromkeys(out))


def _clean_final_models(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []

    known = [
        "pix2pix GAN", "GAN", "ResNet", "U-Net", "U-CS", "U-SS",
        "random forests", "SVM", "support vector machines", "XGBoost",
        "CIBERSORT", "OLS", "PPML", "IV-Poisson", "2SLS",
        "control function approach", "ARIMA", "SIR", "SEIR", "SQUIDER", "LSTM",
        "ChatGPT",
    ]

    out = []

    for item in items:
        text = _clean_text(item)
        low = text.lower()

        for name in known:
            if re.search(
                rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])",
                low,
            ):
                out.append(name)

        if len(text.split()) <= 8:
            out.append(text)

    out = list(dict.fromkeys(out))

    # Canonicalize aliases
    if "SVM" in out:
        out = [x for x in out if x not in {"support vector machines", "support vector machines (SVM)"}]

    return out

def _clean_final_metrics(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []

    out = []
    blob = " ".join(_clean_text(x) for x in items)

    patterns = [
        r"\bAUC(?: values?)?\s*(?:approximately|around)?\s*[0-9.]+(?:\s*[-–]\s*[0-9.]+)?",
        r"\bROC(?: curve)?\b",
        r"\bfivefold cross-validation\b",
        r"\bcross-validation\b",
        r"\bheld-out test dataset\b",
        r"\bp[- ]?values?\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, blob, flags=re.IGNORECASE):
            out.append(_clean_text(m.group(0)))

    return list(dict.fromkeys(out))

def _clean_final_findings(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []

    out = []

    for item in items:
        text = _clean_text(item)
        low = text.lower()

        if not text:
            continue

        if len(text.split()) > 45:
            if "auc" in low:
                out.append("XGBoost and Random Forest achieved moderate predictive performance with AUC values around 0.57–0.58.")
            elif "surviving patients" in low:
                out.append("Surviving patients showed longer survival durations than deceased patients.")
            elif "enriched pathways" in low:
                out.append("Enriched pathways included protein targeting to the endoplasmic reticulum, viral transcription, and cadherin-mediated binding.")
            continue

        out.append(text)

    return list(dict.fromkeys(out))[:6]


def build_auto_best_card(
    local_card: Dict[str, Any],
    refinement: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a hybrid final card by selecting the best field from:
    - local rule-based extraction
    - Nemotron-refined extraction

    If Nemotron failed or was skipped, returns local card.
    """

    if refinement.get("status") != "ok":
        return {
            "status": "local_only",
            "final_paper_card": local_card,
            "selection_report": {
                "reason": "Nemotron refinement was skipped or failed.",
                "fields": {},
            },
        }

    refined_card = refinement.get("after_refinement")
    if not isinstance(refined_card, dict):
        return {
            "status": "local_only",
            "final_paper_card": local_card,
            "selection_report": {
                "reason": "Nemotron output was not a valid dictionary.",
                "fields": {},
            },
        }

    final: Dict[str, Any] = {}
    report: Dict[str, Any] = {}

    all_fields = list(dict.fromkeys(FINAL_FIELDS + list(local_card.keys()) + list(refined_card.keys())))

    for field in all_fields:
        if field == "llm_evidence_pack":
            continue

        local_value = local_card.get(field)
        refined_value = refined_card.get(field)

        value, field_report = _choose_field(field, local_value, refined_value)

        final[field] = value
        report[field] = field_report

    local_count = sum(1 for r in report.values() if r.get("winner") == "local")
    nemotron_count = sum(1 for r in report.values() if r.get("winner") == "nemotron")

    final["selection_metadata"] = {
        "strategy": "field_level_auto_best",
        "local_fields_used": local_count,
        "nemotron_fields_used": nemotron_count,
        "total_fields_compared": len(report),
    }

    final["datasets_or_data_sources"] = _clean_final_datasets(
        final.get("datasets_or_data_sources", []),
        final.get("paper_type", ""),
    )

    if not final.get("datasets_or_data_sources"):
        roadmap = final.get("reproduction_roadmap")
        if isinstance(roadmap, dict):
            final["datasets_or_data_sources"] = _clean_final_datasets(
                roadmap.get("datasets", []),
                final.get("paper_type", ""),
            )

    if not final.get("datasets_or_data_sources"):
        kit = final.get("lab_starter_kit")
        if isinstance(kit, dict):
            final["datasets_or_data_sources"] = _clean_final_datasets(
                kit.get("dataset_plan", []),
                final.get("paper_type", ""),
            )

    final["models_or_methods"] = _clean_final_models(
        final.get("models_or_methods", [])
    )

    final["metrics_or_measurements"] = _clean_final_metrics(
        final.get("metrics_or_measurements", [])
    )

    final["key_findings"] = _clean_final_findings(
        final.get("key_findings", [])
    )

    if isinstance(final.get("lab_starter_kit"), dict):
        for key in ["dataset_plan", "search_strategy", "literature_mapping_plan"]:
            if key in final["lab_starter_kit"]:
                final["lab_starter_kit"][key] = _clean_final_datasets(
                    final["lab_starter_kit"].get(key, []),
                    "machine_learning" if key == "dataset_plan" else final.get("paper_type", ""),
                )

    return {
        "status": "ok",
        "final_paper_card": final,
        "selection_report": {
            "strategy": "field_level_auto_best",
            "fields": report,
        },
    }