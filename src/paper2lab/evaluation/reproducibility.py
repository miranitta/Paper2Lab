"""
reproducibility.py — Missing-information detection and reproducibility scoring.

Scores are heuristic and evidence-based. The goal is to produce a useful local
candidate before Nemotron refinement, while avoiding overconfident scores when
PDF extraction evidence is noisy.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "", text)
    return text.strip(" .;:\n\t")


def _joined_text(extracted: Dict[str, Any]) -> str:
    parts: List[str] = []
    for sec in extracted.get("sections", []) or []:
        if sec.get("role") in {"references", "appendix", "boilerplate"}:
            continue
        parts.append(str(sec.get("title", "")))
        parts.append(str(sec.get("text", "")))
    return _clean("\n".join(parts)).lower()


def _has_any(text: str, terms: List[str]) -> bool:
    return any(t.lower() in text for t in terms)


def _matched_terms(text: str, terms: List[str], limit: int = 5) -> List[str]:
    return [t for t in terms if t.lower() in text][:limit]


# ---------------------------------------------------------------------------
# Paper-type-specific reproducibility checks
# ---------------------------------------------------------------------------

def _check_items(paper_type: str) -> Dict[str, List[str]]:
    if paper_type == "systematic_review":
        return {
            "search databases specified": [
                "pubmed", "scopus", "web of knowledge", "eric", "cochrane",
                "database", "databases",
            ],
            "search date range specified": [
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
                "between", "from", "until", "to january", "published between",
            ],
            "inclusion criteria specified": [
                "inclusion criteria", "eligibility criteria", "eligible studies",
            ],
            "exclusion criteria specified": [
                "exclusion criteria", "excluded", "not being", "were excluded",
            ],
            "screening process specified": [
                "screened", "screening", "titles and abstracts", "two independent",
                "reviewers", "duplicates", "endnote",
            ],
            "quality assessment specified": [
                "quality assessment", "risk of bias", "best evidence medical education",
                "valid tool", "critical appraisal", "assessment tool",
            ],
            "number of included studies specified": [
                "included", "enrolled", "final review", "studies were included",
                "articles were included", "10 articles", "ten studies",
            ],
        }

    if paper_type == "machine_learning":
        return {
            "dataset details specified": [
                "dataset", "training set", "test set", "validation set", "benchmark",
                "corpus", "samples", "instances",
            ],
            "train/validation/test split specified": [
                "train", "validation", "test", "split", "dev set", "development set",
            ],
            "model architecture specified": [
                "architecture", "layers", "encoder", "decoder", "transformer", "cnn",
                "resnet", "bert", "attention", "feed-forward",
            ],
            "hyperparameters specified": [
                "learning rate", "batch size", "epochs", "optimizer", "dropout",
                "weight decay", "warmup", "scheduler",
            ],
            "hardware specified": [
                "gpu", "tpu", "cuda", "p100", "v100", "a100", "nvidia",
            ],
            "evaluation metrics specified": [
                "accuracy", "f1", "auc", "bleu", "rouge", "perplexity", "rmse", "mae",
                "precision", "recall",
            ],
            "code availability specified": [
                "github", "code", "repository", "available at", "source code",
            ],
            "random seed specified": ["random seed", "seed"],
        }

    if paper_type == "clinical_study":
        return {
            "cohort or participants specified": [
                "patients", "participants", "cohort", "subjects", "population",
            ],
            "inclusion criteria specified": ["inclusion criteria", "eligible"],
            "exclusion criteria specified": ["exclusion criteria", "excluded"],
            "outcomes specified": ["outcome", "endpoint", "mortality", "diagnosis"],
            "statistical analysis specified": [
                "statistical analysis", "p-value", "confidence interval", "regression",
            ],
            "ethics approval specified": [
                "ethics", "institutional review", "informed consent", "irb",
            ],
        }

    return {
        "data/source details specified": [
            "data", "dataset", "source", "samples", "studies", "articles",
        ],
        "method/procedure specified": [
            "method", "procedure", "approach", "experiment", "analysis",
        ],
        "evaluation or analysis specified": [
            "evaluation", "result", "metric", "analysis", "measured", "assessed",
        ],
        "limitations discussed": ["limitation", "limitations", "future work"],
    }


# ---------------------------------------------------------------------------
# Evidence quality / noise handling
# ---------------------------------------------------------------------------

_NOISY_EVIDENCE_MARKERS = [
    "the there",
    "being accordingly",
    "endnote teachers",
    "resultsare",
    "analysis of the resultsare",
    "table 2:",
    "department of",
    "university of",
    "medical sciences",
    "corresponding author",
    "access this article online",
    "how to cite",
    "need this systematic review",
    "the that",
]


def _roadmap_blob(paper_card: Dict[str, Any]) -> str:
    roadmap = paper_card.get("reproduction_roadmap") or {}
    parts: List[str] = []

    for key in [
        "datasets",
        "software_requirements",
        "experimental_steps",
        "evaluation_procedure",
        "expected_outputs",
        "missing_for_reproduction",
    ]:
        value = roadmap.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    parts.extend(str(v) for v in item.values())
                else:
                    parts.append(str(item))
        elif value:
            parts.append(str(value))

    return _clean(" ".join(parts)).lower()


def _noise_report(extracted: Dict[str, Any], paper_card: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Return count and examples of noisy evidence markers."""
    blob = _roadmap_blob(paper_card)
    if not blob:
        # Fallback to body text only if roadmap is not yet attached.
        blob = _joined_text(extracted)

    found = [m for m in _NOISY_EVIDENCE_MARKERS if m in blob]

    # Extra generic noise signals.
    if len(re.findall(r"\[\d+", blob)) >= 12:
        found.append("many citation fragments")
    if re.search(r"\b(the|and|of)\s+\1\b", blob):
        found.append("repeated function-word artifact")

    return len(found), found[:8]


def _apply_score_caps(
    paper_type: str,
    score: float,
    missing: List[str],
    extracted: Dict[str, Any],
    paper_card: Dict[str, Any],
) -> Tuple[float, List[str], Dict[str, Any]]:
    """Prevent misleadingly high scores when evidence is noisy or incomplete."""
    diagnostics: Dict[str, Any] = {}
    noise_count, noise_examples = _noise_report(extracted, paper_card)
    diagnostics["noise_count"] = noise_count
    diagnostics["noise_examples"] = noise_examples

    if noise_count > 0:
        msg = "some extracted evidence appears noisy due to PDF layout"
        if msg not in missing:
            missing.append(msg)

    # Systematic reviews should not get 1.0 if roadmap/evidence is visibly noisy.
    if paper_type == "systematic_review":
        if noise_count >= 3:
            score = min(score, 0.65)
        elif noise_count >= 1:
            score = min(score, 0.75)

        roadmap = paper_card.get("reproduction_roadmap") or {}
        if not roadmap.get("experimental_steps"):
            score = min(score, 0.70)
        if not roadmap.get("evaluation_procedure"):
            score = min(score, 0.70)

    # ML papers need either hyperparameters or code/hardware to be strong.
    if paper_type == "machine_learning":
        text = _joined_text(extracted)
        has_hparams = _has_any(text, ["learning rate", "batch size", "optimizer", "dropout", "epoch"])
        has_code = _has_any(text, ["github", "repository", "code available", "source code"])
        has_hardware = _has_any(text, ["gpu", "tpu", "cuda", "p100", "v100", "a100"])
        if not has_hparams:
            score = min(score, 0.80)
        if not has_code and not has_hardware:
            score = min(score, 0.85)

    return round(score, 3), missing, diagnostics


def _score_level(score: float) -> str:
    if score >= 0.80:
        return "strong"
    if score >= 0.50:
        return "partial"
    return "weak"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reproducibility_report(extracted: Dict[str, Any], paper_card: Dict[str, Any]) -> Dict[str, Any]:
    paper_type = paper_card.get("paper_type", "general_research")
    text = _joined_text(extracted)
    checks = _check_items(paper_type)

    detected: List[str] = []
    missing: List[str] = []
    evidence: Dict[str, List[str]] = {}

    for label, terms in checks.items():
        if _has_any(text, terms):
            detected.append(label)
            evidence[label] = _matched_terms(text, terms)
        else:
            missing.append(label)

    # Candidate-card overrides for generic papers.
    if paper_card.get("datasets_or_data_sources") and "data/source details specified" in missing:
        missing.remove("data/source details specified")
        detected.append("data/source details specified")
        evidence["data/source details specified"] = ["paper_card.datasets_or_data_sources"]

    if paper_card.get("metrics_or_measurements") and "evaluation or analysis specified" in missing:
        missing.remove("evaluation or analysis specified")
        detected.append("evaluation or analysis specified")
        evidence["evaluation or analysis specified"] = ["paper_card.metrics_or_measurements"]

    total = max(1, len(checks))
    score = len(detected) / total

    score, missing, diagnostics = _apply_score_caps(
        paper_type=paper_type,
        score=score,
        missing=missing,
        extracted=extracted,
        paper_card=paper_card,
    )

    # Deduplicate while preserving order.
    detected = list(dict.fromkeys(detected))
    missing = list(dict.fromkeys(missing))

    return {
        "paper_type": paper_type,
        "score": score,
        "level": _score_level(score),
        "detected_items": detected,
        "missing_items": missing,
        "evidence_terms": evidence,
        "diagnostics": diagnostics,
    }
