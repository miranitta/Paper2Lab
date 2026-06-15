"""
roadmap.py — Deterministic reproduction-roadmap builder for Paper2Lab.

Input: section-aware extraction dict + optional paper_card.
Output: structured reproduction roadmap candidate.

This module is intentionally local/rule-based. Modal/Nemotron should refine this
later, not replace it.

Design goals:
- Keep the public API stable: build_reproduction_roadmap(extracted, paper_card=None)
- Be paper-type aware: ML papers, systematic reviews, clinical/general papers.
- Avoid noisy merged PDF lines from two-column layouts.
- Keep concise evidence with section/page locations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


# ---------------------------------------------------------------------------
# Marker banks
# ---------------------------------------------------------------------------

_DATASET_MARKERS = [
    "dataset", "data", "corpus", "benchmark", "training set", "test set", "validation set",
    "patients", "samples", "records", "images", "sentences", "tokens", "articles", "studies",
    "pubmed", "scopus", "web of knowledge", "eric", "educational resources and information center",
    "cochrane", "wmt", "imagenet", "cifar", "mnist", "glue", "squad", "penn treebank", "wsj",
]

_KNOWN_DATA_SOURCES = [
    "PubMed",
    "Scopus",
    "Web of Knowledge",
    "ERIC",
    "Educational Resources and Information Center",
    "Cochrane",
    "WMT 2014",
    "WMT",
    "ImageNet",
    "CIFAR-10",
    "CIFAR-100",
    "MNIST",
    "GLUE",
    "SuperGLUE",
    "SQuAD",
    "Penn Treebank",
    "Wall Street Journal",
    "WSJ",
]

_SOFTWARE_MARKERS = [
    "python", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "r", "matlab",
    "cuda", "gpu", "github", "repository", "code", "implementation", "package", "library",
    "endnote", "excel", "spss", "stata", "prisma", "docker",
]

_KNOWN_SOFTWARE = [
    "Python", "PyTorch", "TensorFlow", "Keras", "Scikit-learn", "R", "MATLAB",
    "CUDA", "Docker", "GitHub", "EndNote", "Excel", "SPSS", "Stata", "PRISMA",
]

_EXPERIMENT_MARKERS = [
    "trained", "fine-tuned", "pre-trained", "evaluated", "optimized", "searched", "screened",
    "selected", "included", "excluded", "randomized", "split", "preprocessed", "augmented",
    "we train", "we trained", "we evaluate", "we evaluated", "we search", "we searched",
    "inclusion criteria", "exclusion criteria", "eligibility criteria", "data extraction",
    "titles and abstracts", "duplicate", "endnote", "excel",
]

_EVAL_MARKERS = [
    "accuracy", "precision", "recall", "f1", "auc", "roc", "bleu", "rouge", "perplexity",
    "loss", "rmse", "mae", "statistical", "p-value", "confidence interval", "evaluation",
    "assessed", "measured", "score", "metric", "kirkpatrick", "quality assessment", "meta-analysis",
    "best evidence medical education", "beme", "final review", "included studies",
]

_OUTPUT_MARKERS = [
    "achieved", "achieves", "outperformed", "outperforms", "improved", "score", "accuracy",
    "bleu", "f1", "auc", "included", "selected", "final review", "articles", "studies",
]


# ---------------------------------------------------------------------------
# Cleaning and sentence filtering
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip(" .;:\n\t")


def _is_noisy(sentence: str) -> bool:
    s = _clean(sentence)
    low = s.lower()

    bad_fragments = [
        "corresponding author",
        "how to cite",
        "access this article online",
        "department of",
        "university of",
        "medical sciences",
        "received:",
        "accepted:",
        "published:",
        "copyright",
        "license",
        "all rights reserved",
        "gmail.com",
        "@",
        "table of contents",
        "journal of education and health promotion",
        "endnote teachers",
        "being accordingly",
        "need this systematic review",
        "the that",
        "of the there",
        "the the evidence",
        "table 1:",
        "table 2:",
        "table 3:",
        "table 4:",
    ]
    if any(x in low for x in bad_fragments):
        return True

    if len(s.split()) > 48:
        return True

    if len(re.findall(r"\[\d+", s)) >= 2:
        return True

    if sentence.count("|") >= 2 or sentence.count("%") >= 6:
        return True

    # Many merged two-column artifacts have two unrelated capitalized clauses
    # without a normal sentence boundary.
    if re.search(r"\b(the|this|therefore|besides|fisher)\b.+\b(the|this|therefore|accordingly)\b", low) and len(s.split()) > 34:
        return True

    # Reject strings that look like fragments rather than standalone steps.
    if len(s.split()) < 5:
        return True

    return False


def _split_sentences(text: str) -> List[str]:
    text = _clean(text)
    # Also split before uppercase section labels that PyMuPDF sometimes merges.
    text = re.sub(r"\b(ABSTRACT|INTRODUCTION|MATERIALS AND METHODS|RESULTS|DISCUSSION|CONCLUSION):", r". \1:", text)
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    out: List[str] = []
    for s in raw:
        s = _clean(s)
        if 35 <= len(s) <= 340 and not _is_noisy(s):
            out.append(s)
    return out


def _dedupe_strings(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        clean = _clean(str(item))
        key = re.sub(r"[^a-z0-9]+", " ", clean.lower()).strip()[:180]
        if key and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _dedupe_dicts(items: List[Dict[str, Any]], key_name: str = "text") -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        text = _clean(str(item.get(key_name, "")))
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:180]
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _section_texts(extracted: Dict[str, Any], roles: set[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    blocked_titles = {"front matter", "keywords", "keywords:", "table of contents"}
    paper_title = _clean(extracted.get("title") or "").lower()

    for sec in extracted.get("sections", []):
        role = sec.get("role", "other")
        title = _clean(sec.get("title") or "")
        low_title = title.lower()
        if role not in roles:
            continue
        if low_title in blocked_titles:
            continue
        if paper_title and low_title == paper_title:
            continue
        rows.append(sec)
    return rows


def _find_evidence(
    sections: List[Dict[str, Any]],
    markers: List[str],
    limit: int = 8,
    require_number: bool = False,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    marker_lows = [m.lower() for m in markers]

    for sec in sections:
        title = sec.get("title", "")
        role = sec.get("role", "other")
        for sent in _split_sentences(sec.get("text", "")):
            low = sent.lower()
            if require_number and not re.search(r"\d", sent):
                continue
            if any(m in low for m in marker_lows):
                hits.append({
                    "text": sent,
                    "section": title,
                    "role": role,
                    "page_start": sec.get("page_start"),
                    "page_end": sec.get("page_end"),
                })
                if len(hits) >= limit:
                    return _dedupe_dicts(hits)
    return _dedupe_dicts(hits)


# ---------------------------------------------------------------------------
# Structured extraction helpers
# ---------------------------------------------------------------------------

def _extract_known_sources(text: str) -> List[str]:
    low = text.lower()
    found: List[str] = []
    aliases = {
        "PubMed": ["pubmed"],
        "Scopus": ["scopus"],
        "Web of Knowledge": ["web of knowledge", "thomson reuters"],
        "ERIC": ["eric", "educational resources and information center"],
        "Educational Resources and Information Center": ["educational resources and information center"],
        "Cochrane": ["cochrane"],
        "WMT 2014": ["wmt 2014"],
        "WMT": ["wmt"],
        "ImageNet": ["imagenet"],
        "CIFAR-10": ["cifar-10", "cifar 10"],
        "CIFAR-100": ["cifar-100", "cifar 100"],
        "MNIST": ["mnist"],
        "GLUE": ["glue"],
        "SuperGLUE": ["superglue"],
        "SQuAD": ["squad"],
        "Penn Treebank": ["penn treebank"],
        "Wall Street Journal": ["wall street journal"],
        "WSJ": ["wsj"],
    }
    for canonical, keys in aliases.items():
        if any(k in low for k in keys):
            found.append(canonical)
    return _dedupe_strings(found)


def _extract_software_from_text(text: str) -> List[str]:
    low = text.lower()
    found: List[str] = []
    for name in _KNOWN_SOFTWARE:
        # Avoid false positive: single-letter R appears everywhere, require context.
        if name == "R":
            if re.search(r"\bR\b", text) and any(x in low for x in ["statistical", "analysis", "software", "package"]):
                found.append(name)
            continue
        if name.lower() in low:
            found.append(name)
    return _dedupe_strings(found)


def _extract_count_outputs(text: str) -> List[str]:
    outputs: List[str] = []
    patterns = [
        r"\b(?:totally|overall|in total),?\s*\d+[\w\s-]{0,40}\b(?:articles|studies|records|abstracts|patients|samples)\b",
        r"\b(?:final review|review)\s+(?:included|enrolled)\s+\d+\s+(?:articles|studies)\b",
        r"\b\d+\s+(?:articles|studies|records|abstracts|patients|samples)\s+(?:were|was)\s+(?:selected|included|enrolled|identified)\b",
        r"\bbetween\s+[A-Z][a-z]+\s+\d{4}\s+and\s+[A-Z][a-z]+\s+\d{4}\b",
        r"\bfrom\s+[A-Z][a-z]+\s+\d{4}\s+to\s+[A-Z][a-z]+\s+\d{4}\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            outputs.append(_clean(m.group(0)))
    return _dedupe_strings(outputs)[:8]


def _compact_datasets(paper_card: Dict[str, Any], evidence: List[Dict[str, Any]], all_text: str) -> List[str]:
    datasets: List[str] = []

    # Prefer canonical names over noisy sentences.
    datasets.extend(_extract_known_sources(all_text))

    # Keep short, clean card items.
    for item in paper_card.get("datasets_or_data_sources") or []:
        clean = _clean(item)
        if not clean or _is_noisy(clean):
            continue
        if len(clean.split()) <= 12:
            datasets.append(clean)

    # Add concise evidence only if it is clean and informative.
    for hit in evidence:
        clean = _clean(hit.get("text", ""))
        if not clean or _is_noisy(clean):
            continue
        if len(clean.split()) <= 24:
            datasets.append(clean)

    return _dedupe_strings(datasets)[:10]


def _roadmap_level(missing_count: int, detected_count: int, noisy_evidence_count: int = 0) -> str:
    # Be less overconfident when evidence is present but noisy.
    if noisy_evidence_count >= 4:
        return "partial" if detected_count >= 5 else "weak"
    if detected_count >= 8 and missing_count <= 2:
        return "strong"
    if detected_count >= 5 and missing_count <= 4:
        return "partial"
    return "weak"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reproduction_roadmap(
    extracted: Dict[str, Any],
    paper_card: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a local, evidence-grounded reproduction roadmap candidate."""
    paper_card = paper_card or {}
    paper_type = paper_card.get("paper_type", "general_research")

    method_sections = _section_texts(extracted, {"methodology", "experiments"})
    result_sections = _section_texts(extracted, {"results", "discussion", "conclusion"})
    all_body_sections = _section_texts(
        extracted,
        {"methodology", "experiments", "results", "discussion", "conclusion", "introduction", "abstract"},
    )

    all_text = "\n".join(sec.get("text", "") for sec in all_body_sections)
    method_text = "\n".join(sec.get("text", "") for sec in method_sections)

    dataset_evidence = _find_evidence(all_body_sections, _DATASET_MARKERS, limit=12)
    software = _extract_software_from_text(all_text)
    experimental_steps = _find_evidence(method_sections or all_body_sections, _EXPERIMENT_MARKERS, limit=10)
    evaluation = _find_evidence(result_sections + method_sections, _EVAL_MARKERS, limit=10)

    datasets = _compact_datasets(paper_card, dataset_evidence, all_text)

    expected_outputs: List[str] = []
    for item in paper_card.get("metrics_or_measurements", [])[:5]:
        clean = _clean(item)
        if clean and not _is_noisy(clean):
            expected_outputs.append(clean)
    for item in paper_card.get("key_findings", [])[:5]:
        clean = _clean(item)
        if clean and not _is_noisy(clean):
            expected_outputs.append(clean)
    expected_outputs.extend(_extract_count_outputs(all_text))
    expected_outputs = _dedupe_strings(expected_outputs)[:8]

    missing: List[str] = []
    if not datasets:
        missing.append("dataset or source corpus details are missing")
    if not experimental_steps:
        missing.append("experimental or procedural steps are missing")
    if not evaluation:
        missing.append("evaluation procedure is missing")
    if paper_type == "machine_learning" and not software:
        missing.append("software/framework requirements are missing")
    if paper_type == "systematic_review":
        low = method_text.lower() or all_text.lower()
        if not any(x in low for x in ["inclusion criteria", "eligibility criteria"]):
            missing.append("inclusion criteria are missing")
        if "exclusion criteria" not in low:
            missing.append("exclusion criteria are missing")
        if not any(x in low for x in ["quality assessment", "risk of bias", "best evidence medical education", "valid tool"]):
            missing.append("quality assessment method is missing")

    noisy_count = 0
    for item in experimental_steps + evaluation + dataset_evidence:
        if _is_noisy(item.get("text", "")):
            noisy_count += 1

    detected_count = len(datasets) + len(software) + len(experimental_steps) + len(evaluation) + len(expected_outputs)

    return {
        "paper_type": paper_type,
        "datasets": datasets,
        "software_requirements": software,
        "experimental_steps": [
            {
                "step": i + 1,
                "description": x["text"],
                "section": x["section"],
                "page_start": x.get("page_start"),
                "page_end": x.get("page_end"),
            }
            for i, x in enumerate(experimental_steps[:10])
        ],
        "evaluation_procedure": [
            {
                "description": x["text"],
                "section": x["section"],
                "page_start": x.get("page_start"),
                "page_end": x.get("page_end"),
            }
            for x in evaluation[:10]
        ],
        "expected_outputs": expected_outputs,
        "missing_for_reproduction": missing,
        "estimated_reproducibility": _roadmap_level(len(missing), detected_count, noisy_count),
    }
