"""
pdf_loader.py — Paper2Lab universal section-aware PDF ingestion.

Final extraction-layer guarantees:
- Field-agnostic section detection across ML, NLP, CV, biomedical, physics,
  education, social-science, economics, and interdisciplinary papers.
- Multi-signal heading confidence scoring: lexical headings, numbered/roman
  headings, ALL-CAPS headings, font size, bold text, and vertical spacing.
- General anti-noise rules: boilerplate, metric-only headings, table-cell-like
  headings, reference items, and fragmented tables.
- Clean body text excludes References/Bibliography, Appendix/Supplementary, and
  boilerplate so downstream extraction does not get polluted.
- raw_text and all_sections are preserved for traceability/debugging.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF


class PDFIngestionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Section taxonomy
# ---------------------------------------------------------------------------

SECTION_KEYWORDS: Set[str] = {
    # Universal/front matter
    "abstract", "keywords", "introduction", "background", "overview",
    "motivation", "problem statement", "problem formulation",

    # Related work / literature
    "related work", "related works", "prior work", "prior art",
    "literature review", "literature survey", "state of the art",

    # Theory / framework
    "theoretical background", "theory", "framework", "preliminaries",
    "notation", "problem definition", "mathematical background",

    # Methodology
    "materials and methods", "material and methods", "methods", "method",
    "methodology", "approach", "proposed approach", "proposed method",
    "model", "model architecture", "architecture", "system design",
    "system overview", "implementation", "implementation details",
    "technical details", "training", "training details", "training procedure",
    "experimental setup", "experimental settings", "experimental design",
    "data collection", "data preprocessing", "data preparation",
    "study design", "participants", "procedure",

    # Experiments/results
    "experiments", "experiment", "evaluation", "evaluations",
    "results", "results and discussion", "results and analysis",
    "empirical evaluation", "empirical results", "analysis",
    "quantitative analysis", "qualitative analysis", "ablation",
    "ablation study", "ablation studies", "case study", "case studies",

    # Discussion/limitations/conclusion
    "discussion", "limitations", "limitation", "future work",
    "future directions", "conclusion", "conclusions", "summary",
    "concluding remarks",

    # Admin/back matter
    "data availability", "code availability", "availability",
    "ethics statement", "ethical considerations", "ethical approval",
    "acknowledgements", "acknowledgments", "funding",
    "conflict of interest", "competing interests", "author contributions",
    "references", "bibliography", "works cited", "literature cited",
    "appendix", "appendices", "supplementary material",
    "supplementary information", "supplemental material", "supplementary",
    "supplemental",
}

REFERENCE_SECTION_NAMES: Set[str] = {
    "references", "bibliography", "works cited", "literature cited",
}

APPENDIX_SECTION_NAMES: Set[str] = {
    "appendix", "appendices", "supplementary material",
    "supplementary information", "supplemental material", "supplementary",
    "supplemental",
}

BOILERPLATE_SECTION_NAMES: Set[str] = {
    "acknowledgements", "acknowledgments", "author contributions",
    "competing interests", "conflict of interest", "additional information",
    "publisher's note", "open access", "correspondence",
    "reprints and permissions", "funding", "ethics statement",
    "ethical considerations", "ethical approval", "data availability",
    "code availability", "availability",
}

ROLE_ALIASES: Dict[str, List[str]] = {
    "front_matter": ["front matter"],
    "abstract": ["abstract"],
    "keywords": ["keywords"],
    "introduction": [
        "introduction", "overview", "motivation", "problem statement",
        "problem formulation",
    ],
    "related_work": ["related work", "related works", "prior work", "prior art"],
    "background": ["background", "preliminaries", "notation"],
    "theory": [
        "theoretical background", "theory", "framework",
        "mathematical background", "problem definition",
    ],
    "methodology": [
        "materials and methods", "material and methods", "methods", "method",
        "methodology", "approach", "proposed approach", "proposed method",
        "model", "model architecture", "architecture", "system design",
        "system overview", "implementation", "implementation details",
        "technical details", "training", "training details", "training procedure",
        "experimental setup", "experimental settings", "experimental design",
        "data collection", "data preprocessing", "data preparation",
        "study design", "participants", "procedure",
    ],
    "experiments": [
        "experiments", "experiment", "evaluation", "evaluations",
        "empirical evaluation", "ablation", "ablation study", "ablation studies",
        "case study", "case studies",
    ],
    "results": [
        "results", "results and discussion", "results and analysis",
        "empirical results", "quantitative analysis", "qualitative analysis",
        "analysis",
    ],
    "discussion": ["discussion"],
    "limitations": ["limitations", "limitation"],
    "future_work": ["future work", "future directions"],
    "conclusion": ["conclusion", "conclusions", "summary", "concluding remarks"],
    "references": list(REFERENCE_SECTION_NAMES),
    "appendix": list(APPENDIX_SECTION_NAMES),
    "boilerplate": list(BOILERPLATE_SECTION_NAMES),
}

BOILERPLATE_PHRASES: List[str] = [
    "provided proper attribution", "google hereby grants permission",
    "permission to reproduce", "all rights reserved", "copyright ©",
    "under the terms of", "creative commons", "open access article",
    "preprint server", "arxiv:", "doi:", "received:", "accepted:",
    "published online", "correspondence to",
]

METRIC_ONLY_TERMS: Set[str] = {
    "accuracy", "acc", "precision", "recall", "sensitivity", "specificity",
    "f1", "f1 score", "auc", "roc auc", "auroc", "auprc", "bleu",
    "rouge", "rouge-l", "meteor", "map", "ndcg", "wer", "cer",
    "perplexity", "ppl", "loss", "rmse", "mae", "mse", "iou",
    "dice", "ari", "nmi", "top-1", "top-5", "er", "hr",
}

TABLE_CELL_HINTS: List[str] = [
    "baseline", "ours", "oracle", "ensemble", "single model", "dev", "test",
    "train", "training", "validation", "params", "flops", "gpu", "cpu",
    "memory", "latency",
]

_HEADING_CONFIDENCE_THRESHOLD = 0.55
_MIN_SECTION_WORDS = 20


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = (
        text.replace("\x00", " ")
        .replace("\u00a0", " ")
        .replace("\u000f", " ")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_heading(text: str) -> str:
    text = _clean_text(text).strip()
    text = re.sub(r"^[#*\s]+", "", text)
    text = re.sub(r"^(section|chapter|part)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[IVXLCDM]+[.)]\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[A-Z][.)]\s+", "", text)
    text = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", text)
    text = re.sub(r"[:.\s]+$", "", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def _heading_level(text: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)", text.strip())
    if match:
        return min(6, match.group(1).count(".") + 1)
    return 1


def _role_for_title(title: str) -> str:
    norm = _normalize_heading(title)
    for role, aliases in ROLE_ALIASES.items():
        if norm in aliases:
            return role
    for role, aliases in ROLE_ALIASES.items():
        if any(alias in norm for alias in aliases):
            return role
    return "other"


def _line_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_heading(text)).strip()


# ---------------------------------------------------------------------------
# Low-level PyMuPDF line extraction
# ---------------------------------------------------------------------------

def _span_is_bold(span: Dict[str, Any]) -> bool:
    font = span.get("font", "") or ""
    flags = int(span.get("flags", 0) or 0)
    return "bold" in font.lower() or "heavy" in font.lower() or bool(flags & 16)


def _extract_page_lines(page: fitz.Page, page_number: int) -> List[Dict[str, Any]]:
    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    page_width = float(page.rect.width)
    mid_x = page_width / 2.0
    raw: List[Dict[str, Any]] = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_no = block.get("number", 0)
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if (s.get("text") or "").strip()]
            if not spans:
                continue
            text = _clean_text("".join(s.get("text", "") for s in spans))
            if not text:
                continue

            sizes = [round(float(s.get("size", 0.0)), 1) for s in spans if s.get("size")]
            max_size = max(sizes) if sizes else 0.0
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0
            bold_ratio = sum(1 for s in spans if _span_is_bold(s)) / max(1, len(spans))
            bbox = tuple(line.get("bbox") or block.get("bbox") or (0, 0, 0, 0))
            x0 = float(bbox[0])
            y0 = float(bbox[1])
            col = 0 if x0 < mid_x else 1

            raw.append({
                "text": text,
                "page_number": page_number,
                "block_no": block_no,
                "bbox": bbox,
                "x0": x0,
                "y0": y0,
                "max_size": max_size,
                "avg_size": avg_size,
                "bold": bold_ratio >= 0.5,
                "span_count": len(spans),
                "col": col,
            })

    if raw:
        max_y = max(float(l["y0"]) for l in raw)
        band_height = max(max_y / 20.0, 1.0)
        raw.sort(key=lambda l: (round(float(l["y0"]) / band_height), int(l["col"]), float(l["y0"])))

    return raw


def _dominant_body_size(lines: List[Dict[str, Any]]) -> float:
    sizes: List[float] = []
    for line in lines:
        size = float(line.get("avg_size") or line.get("max_size") or 0.0)
        words = len(str(line.get("text", "")).split())
        if 7.0 <= size <= 16.0 and words >= 4:
            sizes.append(round(size, 1))
    return Counter(sizes).most_common(1)[0][0] if sizes else 10.0


# ---------------------------------------------------------------------------
# Heading scoring
# ---------------------------------------------------------------------------

def _boilerplate_line(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(phrase in low for phrase in BOILERPLATE_PHRASES)


def _metric_only_heading(text: str) -> bool:
    norm = _normalize_heading(text)
    compact = re.sub(r"[^a-z0-9]+", " ", norm).strip()
    if compact in METRIC_ONLY_TERMS:
        return True

    metric_re = r"\b(" + "|".join(re.escape(m) for m in sorted(METRIC_ONLY_TERMS, key=len, reverse=True)) + r")\b"
    if len(text.split()) <= 5 and re.search(metric_re, compact, flags=re.IGNORECASE):
        if not any(w in compact for w in ["results", "evaluation", "experiment", "analysis", "method"]):
            return True

    letters = re.sub(r"[^A-Za-z]", "", text)
    if 2 <= len(letters) <= 10 and letters.isupper() and len(text.split()) <= 3:
        return True
    return False


def _probable_table_cell(text: str) -> bool:
    raw = _clean_text(text)
    low = raw.lower()
    words = raw.split()
    if not raw or len(words) > 9:
        return False

    if re.search(r"\[\d+\]", raw) and len(words) <= 7:
        return True

    chars = [c for c in raw if not c.isspace()]
    if chars:
        symbol_digit_ratio = sum(1 for c in chars if c.isdigit() or c in ".,±+-×*/=()%[]") / len(chars)
        if symbol_digit_ratio >= 0.35 and len(words) <= 6:
            return True

    if len(words) <= 5 and any(h in low for h in TABLE_CELL_HINTS):
        if not any(sw in low for sw in ["result", "method", "experiment", "evaluation", "discussion", "conclusion"]):
            return True

    if len(words) <= 5 and re.search(r"\b[A-Z][A-Za-z0-9-]*\s*(\+|/|vs\.?|&|×)\s*[A-Z]", raw):
        return True

    return False


def _looks_like_reference_item(line: str) -> bool:
    return bool(re.match(r"^(\[\d+\]|\d+[.)])\s+", line.strip())) and len(line.split()) > 6


_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLCDM]+[.)]|[A-Z][.)])\s+"
    r"[A-Z\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F ,/()\-–—:&']+$",
    re.IGNORECASE,
)


def _heading_confidence(
    line: Dict[str, Any],
    body_size: float,
    prev_line: Optional[Dict[str, Any]],
    next_line: Optional[Dict[str, Any]],
) -> float:
    text = _clean_text(line.get("text", ""))
    if not text:
        return 0.0
    if _boilerplate_line(text) or _metric_only_heading(text) or _probable_table_cell(text):
        return 0.0
    if _looks_like_reference_item(text):
        return 0.0
    if len(text) > 150 or len(text.split()) > 18:
        return 0.0
    if text.endswith(",") or text.endswith(";"):
        return 0.0

    norm = _normalize_heading(text)
    if not norm or norm in {"figure", "table", "fig", "eq", "equation"}:
        return 0.0

    score = 0.0

    if norm in SECTION_KEYWORDS:
        score += 0.70
    else:
        for keyword in SECTION_KEYWORDS:
            if keyword in norm and len(keyword) >= 6:
                score += 0.30
                break

    if _NUMBERED_HEADING_RE.match(text) and len(text.split()) <= 14:
        score += 0.55

    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) >= 4 and letters.isupper() and len(text.split()) <= 10:
        score += 0.40

    size = float(line.get("max_size") or line.get("avg_size") or body_size)
    size_ratio = size / body_size if body_size else 1.0
    if size_ratio >= 1.25:
        score += 0.35
    elif size_ratio >= 1.10:
        score += 0.20

    if line.get("bold"):
        score += 0.20

    bbox = line.get("bbox") or (0, 0, 0, 0)
    y0 = float(bbox[1])
    y1 = float(bbox[3])
    prev_gap = 999.0
    next_gap = 999.0
    if prev_line and prev_line.get("page_number") == line.get("page_number"):
        prev_gap = y0 - float((prev_line.get("bbox") or (0, 0, 0, 0))[3])
    if next_line and next_line.get("page_number") == line.get("page_number"):
        next_gap = float((next_line.get("bbox") or (0, 0, 0, 0))[1]) - y1
    if prev_gap >= 3.0 or next_gap >= 2.0:
        score += 0.15

    if re.match(r"^[A-Z\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F ,/()\-–—:&']+$", text):
        if not re.search(r"\b(the|and|or|but|because|while|which|that)\b.+\.$", text.lower()):
            score += 0.10

    if len(text.split()) < 2 and score < 0.70:
        score *= 0.40

    return min(score, 1.0)


def _detect_heading_keys(lines: List[Dict[str, Any]], body_size: float) -> Set[str]:
    candidate_scores: Dict[str, List[float]] = defaultdict(list)

    for i, line in enumerate(lines):
        prev_line = lines[i - 1] if i else None
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        confidence = _heading_confidence(line, body_size, prev_line, next_line)
        if confidence > 0.0:
            key = _line_key(line["text"])
            if key:
                candidate_scores[key].append(confidence)

    accepted: Set[str] = set()
    for key, scores in candidate_scores.items():
        best = max(scores)
        if best >= _HEADING_CONFIDENCE_THRESHOLD:
            accepted.add(key)
        elif best >= 0.35 and len(scores) >= 2:
            accepted.add(key)
    return accepted


# ---------------------------------------------------------------------------
# Section splitting and back-matter separation
# ---------------------------------------------------------------------------

def _split_sections_from_lines(lines: List[Dict[str, Any]], heading_keys: Set[str]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current_title = "Front Matter"
    current_level = 1
    current_role = "front_matter"
    current_lines: List[str] = []
    page_start: Optional[int] = lines[0]["page_number"] if lines else None
    page_end: Optional[int] = page_start

    def flush() -> None:
        nonlocal current_lines, current_title, current_level, current_role, page_start, page_end
        text = _clean_text("\n".join(current_lines))
        if text:
            sections.append({
                "title": current_title,
                "text": text,
                "level": current_level,
                "role": current_role,
                "page_start": page_start,
                "page_end": page_end,
                "word_count": len(text.split()),
            })
        current_lines = []

    for line in lines:
        text = _clean_text(line["text"])
        key = _line_key(text)
        if key in heading_keys:
            flush()
            current_title = text
            current_level = _heading_level(text)
            current_role = _role_for_title(text)
            page_start = line.get("page_number")
            page_end = page_start
        else:
            current_lines.append(text)
            page_end = line.get("page_number")

    flush()

    merged: List[Dict[str, Any]] = []
    for sec in sections:
        norm = _normalize_heading(sec["title"])
        words = int(sec.get("word_count", len(sec.get("text", "").split())))
        if sec["title"] != "Front Matter" and norm not in SECTION_KEYWORDS and words < _MIN_SECTION_WORDS and merged:
            merged[-1]["text"] = _clean_text(merged[-1]["text"] + "\n" + sec["title"] + "\n" + sec["text"])
            merged[-1]["page_end"] = sec.get("page_end") or merged[-1].get("page_end")
            merged[-1]["word_count"] = len(merged[-1]["text"].split())
        else:
            merged.append(sec)
    return merged


def _separate_back_matter(sections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, str, str]:
    body: List[Dict[str, Any]] = []
    references_parts: List[str] = []
    appendix_parts: List[str] = []
    boilerplate_parts: List[str] = []

    mode = "body"
    for sec in sections:
        role = sec.get("role") or _role_for_title(sec.get("title", ""))
        norm = _normalize_heading(sec.get("title", ""))
        packed = _clean_text(f'{sec.get("title", "")}\n{sec.get("text", "")}')

        if role == "references" or norm in REFERENCE_SECTION_NAMES:
            mode = "references"
        elif role == "appendix" or norm in APPENDIX_SECTION_NAMES:
            if mode != "references":
                mode = "appendix"
        elif role == "boilerplate" or norm in BOILERPLATE_SECTION_NAMES:
            if mode not in {"references", "appendix"}:
                mode = "boilerplate"
        elif mode == "boilerplate":
            mode = "body"

        if mode == "references":
            references_parts.append(packed)
        elif mode == "appendix":
            appendix_parts.append(packed)
        elif mode == "boilerplate":
            boilerplate_parts.append(packed)
        else:
            body.append(sec)

    return body, "\n\n".join(references_parts), "\n\n".join(appendix_parts), "\n\n".join(boilerplate_parts)


# ---------------------------------------------------------------------------
# Title, abstract, references, captions, tables
# ---------------------------------------------------------------------------

_BAD_TITLE_MARKERS = [
    "provided proper attribution", "google hereby grants permission", "arxiv",
    "preprint", "license", "copyright", "all rights reserved",
]

_BLOCKED_TITLE_FRAGMENTS = [
    "vol.", "http", "www.", "received", "accepted", "department", "university",
    "hospital", "college", "institute", "email", "@", "proceedings",
    "journal of", "conference on",
]


def _extract_title(lines: List[Dict[str, Any]]) -> Optional[str]:
    first_page = [line for line in lines if line.get("page_number") == 1]
    candidates: List[Dict[str, Any]] = []

    for line in first_page[:100]:
        text = _clean_text(line["text"])
        low = text.lower()
        if len(text) < 8 or len(text) > 200:
            continue
        if any(x in low for x in _BAD_TITLE_MARKERS + _BLOCKED_TITLE_FRAGMENTS):
            continue
        if re.match(r"^\d+$", text) or _normalize_heading(text) in SECTION_KEYWORDS:
            continue
        if text.endswith(".") and len(text.split()) < 10:
            continue
        candidates.append(line)

    if not candidates:
        return None

    max_size = max(float(c.get("max_size") or 0) for c in candidates)
    title_lines = [c for c in candidates if float(c.get("max_size") or 0) >= max_size - 0.6]

    by_block: Dict[int, List[str]] = defaultdict(list)
    for line in title_lines:
        by_block[int(line.get("block_no", 0))].append(line["text"])

    for _, parts in sorted(by_block.items()):
        title = _clean_text(" ".join(parts))
        if 3 <= len(title.split()) <= 35:
            return title

    for line in candidates[:25]:
        text = _clean_text(line["text"])
        if 3 <= len(text.split()) <= 30 and not text.endswith("."):
            return text
    return None


_ABSTRACT_MARKERS = [
    "we propose", "we present", "we introduce", "we developed", "we investigate",
    "this paper", "this work", "this study", "this article", "here we",
    "in this paper", "in this work", "in this study",
]


def _extract_abstract_from_sections(sections: List[Dict[str, Any]], clean_text: str) -> Optional[str]:
    for sec in sections:
        if sec.get("role") == "abstract" or _normalize_heading(sec.get("title", "")) == "abstract":
            text = _clean_text(sec.get("text", ""))
            text = re.sub(r"^abstract[:\s—–-]*", "", text, flags=re.IGNORECASE).strip()
            if len(text.split()) >= 30:
                return text

    explicit = re.search(
        r"\babstract\b[:\s—–-]*(.{100,2500}?)(?=\n\s*(?:\d+[.)]?\s*)?(?:keywords|introduction|background|1[.\s]|i[.\s])\b)",
        clean_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explicit:
        candidate = _clean_text(explicit.group(1))
        if len(candidate.split()) >= 30:
            return candidate

    paragraphs = [_clean_text(p) for p in re.split(r"\n\s*\n", clean_text[:8000]) if _clean_text(p)]
    for para in paragraphs[:10]:
        low = para.lower()
        if len(para.split()) >= 40 and any(marker in low for marker in _ABSTRACT_MARKERS):
            return para
    return None


_REF_SPLIT_RE = re.compile(
    r"\n(?=\s*(?:\[\d+\]|\d+[.)]|[A-Z][a-zA-Z\-]+,\s+[A-Z]|\(\d{4}\)))"
)


def _parse_references(references_text: str) -> List[str]:
    references_text = _clean_text(references_text)
    if not references_text:
        return []
    references_text = re.sub(r"^references\s*", "", references_text, flags=re.IGNORECASE).strip()
    parts = _REF_SPLIT_RE.split(references_text)

    refs: List[str] = []
    for part in parts:
        ref = _clean_text(part)
        if len(ref) >= 25 and _normalize_heading(ref) not in BOILERPLATE_SECTION_NAMES:
            refs.append(ref)

    seen: Set[str] = set()
    unique: List[str] = []
    for ref in refs:
        key = re.sub(r"\s+", " ", ref.lower())[:160]
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return unique[:300]


_CAPTION_RE = re.compile(
    r"^(?P<label>"
    r"(?:figure|fig\.?|table|tbl\.?|scheme|supplementary figure|supp\.?\s*fig\.?|"
    r"extended data figure|algorithm|listing)\s*[\dIVXLC]+[A-Za-z]?"
    r")[\.:：\s\-–—]*(?P<caption>.+)$",
    re.IGNORECASE,
)


def _extract_captions(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    captions: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    texts = [line["text"] for line in lines]

    for i, line in enumerate(lines):
        m = _CAPTION_RE.match(_clean_text(line["text"]))
        if not m:
            continue
        label = m.group("label").strip()
        caption = m.group("caption").strip()
        if re.match(r"^(shows|illustrates|presents|depicts|demonstrates)\b", caption.lower()):
            continue

        continuation: List[str] = []
        for next_text in texts[i + 1:i + 6]:
            nt = _clean_text(next_text)
            if _CAPTION_RE.match(nt) or _normalize_heading(nt) in SECTION_KEYWORDS:
                break
            if len(nt.split()) < 4:
                break
            continuation.append(nt)
        if continuation:
            caption = _clean_text(caption + " " + " ".join(continuation))

        key = re.sub(r"[^a-z0-9]+", "", label.lower())
        if key in seen:
            continue
        seen.add(key)
        captions.append({
            "label": label,
            "caption": caption,
            "page_number": line.get("page_number"),
        })
    return captions


def _is_valid_table(data: Any) -> bool:
    if not isinstance(data, list) or len(data) < 2:
        return False
    rows = [row for row in data if isinstance(row, list)]
    if len(rows) < 2:
        return False
    cols = max((len(row) for row in rows), default=0)
    if cols < 2:
        return False

    cells = [str(cell).strip() for row in rows for cell in row if cell is not None and str(cell).strip()]
    non_empty = len(cells)
    total_cells = sum(len(row) for row in rows)
    if non_empty < 6:
        return False

    flat = " ".join(cells).lower()
    if non_empty <= 8 and any(noise in flat for noise in ["www.", "http", "vol.", "doi"]):
        return False

    fill_ratio = non_empty / max(total_cells, 1)
    avg_cell_len = sum(len(cell) for cell in cells) / max(non_empty, 1)
    one_char_ratio = sum(1 for cell in cells if len(cell) <= 1) / max(non_empty, 1)
    very_short_ratio = sum(1 for cell in cells if len(cell) <= 2) / max(non_empty, 1)

    if cols > 12 and (fill_ratio < 0.35 or avg_cell_len < 3.0 or one_char_ratio > 0.35):
        return False
    if avg_cell_len < 2.5 and very_short_ratio > 0.60:
        return False
    if cols > 20 and non_empty < 80:
        return False

    meaningful_rows = sum(
        1 for row in rows
        if len([str(c).strip() for c in row if c is not None and str(c).strip()]) >= 2
        and len(" ".join(str(c) for c in row)) >= 12
    )
    return meaningful_rows >= 2


def _quality_report(result: Dict[str, Any]) -> Dict[str, Any]:
    title = result.get("title") or ""
    abstract = result.get("abstract") or ""
    sections = result.get("sections", [])
    roles = {section.get("role", "other") for section in sections}
    n_refs = len(result.get("references", []))

    score = 0.0
    score += 0.15 if len(title.split()) >= 4 else 0.0
    score += 0.20 if len(abstract.split()) >= 30 else 0.0
    score += min(0.20, len(sections) * 0.022)
    score += 0.10 if "methodology" in roles else 0.0
    score += 0.08 if {"results", "experiments"} & roles else 0.0
    score += 0.07 if "conclusion" in roles else 0.0
    score += 0.08 if n_refs >= 5 else (0.03 if n_refs else 0.0)
    score += 0.05 if len(result.get("clean_text", "")) > 4000 else 0.0
    score += 0.04 if result.get("captions") else 0.0
    score += 0.03 if result.get("tables") else 0.0

    if len(abstract.split()) < 20:
        score -= 0.10
    if not sections:
        score -= 0.20

    return {
        "title_found": len(title.split()) >= 4,
        "abstract_found": len(abstract.split()) >= 30,
        "num_sections": len(sections),
        "section_roles": sorted(roles),
        "methodology_section_found": "methodology" in roles,
        "num_references": n_refs,
        "num_captions": len(result.get("captions", [])),
        "num_tables": len(result.get("tables", [])),
        "references_removed_from_clean_text": bool(result.get("references_text")),
        "appendix_removed_from_clean_text": bool(result.get("appendix_text")),
        "quality_score": round(max(0.0, min(score, 0.98)), 3),
    }


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def extract_with_pymupdf(pdf_path: str | Path) -> Dict[str, Any]:
    path = Path(pdf_path)
    if not path.exists():
        raise PDFIngestionError(f"PDF not found: {path}")

    doc = fitz.open(path)
    try:
        all_lines: List[Dict[str, Any]] = []
        pages: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            page_lines = _extract_page_lines(page, page_number)
            all_lines.extend(page_lines)
            page_text = _clean_text("\n".join(line["text"] for line in page_lines))
            pages.append({"page_number": page_number, "text": page_text})

            try:
                found = page.find_tables()
                for table_index, table in enumerate(found.tables):
                    data = table.extract()
                    if _is_valid_table(data):
                        tables.append({
                            "page_number": page_number,
                            "table_index": table_index,
                            "data": data,
                            "engine": "pymupdf",
                            "caption": None,
                        })
            except Exception:
                pass

        full_text = _clean_text("\n\n".join(page["text"] for page in pages))
        body_size = _dominant_body_size(all_lines)
        heading_keys = _detect_heading_keys(all_lines, body_size)
        sections_all = _split_sections_from_lines(all_lines, heading_keys)
        body_sections, references_text, appendix_text, boilerplate_text = _separate_back_matter(sections_all)
        clean_text = _clean_text("\n\n".join(f'{s["title"]}\n{s["text"]}' for s in body_sections))

        title = _extract_title(all_lines)
        abstract = _extract_abstract_from_sections(body_sections, clean_text)
        references = _parse_references(references_text)
        captions = _extract_captions(all_lines)

        # Attach table captions when labels and table indices align.
        cap_map: Dict[str, str] = {}
        for cap in captions:
            label = cap.get("label", "").lower()
            if "table" in label or "tbl" in label:
                key = re.sub(r"[^a-z0-9]", "", label)
                cap_map[key] = cap.get("caption", "")
        for table in tables:
            idx_str = str(table.get("table_index", ""))
            for key, caption in cap_map.items():
                if idx_str in key or key.endswith(idx_str):
                    table["caption"] = caption
                    break

        result: Dict[str, Any] = {
            "source_pdf": path.name,
            "num_pages": len(doc),
            "title": title,
            "abstract": abstract,
            "text": clean_text,
            "clean_text": clean_text,
            "raw_text": full_text,
            "pages": pages,
            "sections": body_sections,
            "all_sections": sections_all,
            "references": references,
            "references_text": references_text,
            "appendix_text": appendix_text,
            "boilerplate_text": boilerplate_text,
            "captions": captions,
            "tables": tables,
            "metadata": {
                "body_font_size": body_size,
                "heading_count": len(heading_keys),
                "removed_back_matter": {
                    "references": bool(references_text),
                    "appendix": bool(appendix_text),
                    "boilerplate": bool(boilerplate_text),
                },
            },
            "extraction_engine": "pymupdf-section-aware-final",
        }
        result["quality"] = _quality_report(result)
        return result
    finally:
        doc.close()


def extract_with_docling(pdf_path: str | Path) -> Dict[str, Any]:
    path = Path(pdf_path)
    if not path.exists():
        raise PDFIngestionError(f"PDF not found: {path}")

    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise PDFIngestionError("Docling is not installed. Run: pip install docling") from exc

    converter = DocumentConverter()
    doc = converter.convert(str(path)).document
    markdown = _clean_text(doc.export_to_markdown())

    fake_lines: List[Dict[str, Any]] = []
    for i, raw in enumerate(markdown.splitlines()):
        is_heading = raw.startswith("#")
        text = _clean_text(re.sub(r"^#+\s*", "", raw))
        if text:
            fake_lines.append({
                "text": text,
                "page_number": None,
                "block_no": i,
                "bbox": (0, i * 10, 0, i * 10 + 8),
                "max_size": 13.0 if is_heading else 10.0,
                "avg_size": 13.0 if is_heading else 10.0,
                "bold": is_heading,
                "col": 0,
            })

    heading_keys = _detect_heading_keys(fake_lines, 10.0)
    for line in fake_lines:
        if line.get("bold") and _normalize_heading(line["text"]) in SECTION_KEYWORDS:
            heading_keys.add(_line_key(line["text"]))

    sections_all = _split_sections_from_lines(fake_lines, heading_keys)
    body_sections, references_text, appendix_text, boilerplate_text = _separate_back_matter(sections_all)
    clean_text = _clean_text("\n\n".join(f'{s["title"]}\n{s["text"]}' for s in body_sections))

    result: Dict[str, Any] = {
        "source_pdf": path.name,
        "num_pages": None,
        "title": _extract_title(fake_lines),
        "abstract": _extract_abstract_from_sections(body_sections, clean_text),
        "text": clean_text,
        "clean_text": clean_text,
        "raw_text": markdown,
        "pages": [],
        "sections": body_sections,
        "all_sections": sections_all,
        "references": _parse_references(references_text),
        "references_text": references_text,
        "appendix_text": appendix_text,
        "boilerplate_text": boilerplate_text,
        "captions": [],
        "tables": [],
        "metadata": {
            "body_font_size": 10.0,
            "heading_count": len(heading_keys),
            "removed_back_matter": {
                "references": bool(references_text),
                "appendix": bool(appendix_text),
                "boilerplate": bool(boilerplate_text),
            },
        },
        "extraction_engine": "docling-section-aware-final",
    }
    result["quality"] = _quality_report(result)
    return result


def extract_pdf(pdf_path: str | Path, engine: str = "pymupdf") -> Dict[str, Any]:
    engine = engine.lower().strip()
    if engine == "pymupdf":
        return extract_with_pymupdf(pdf_path)
    if engine == "docling":
        return extract_with_docling(pdf_path)
    if engine == "auto":
        try:
            pymupdf_result = extract_with_pymupdf(pdf_path)
            try:
                docling_result = extract_with_docling(pdf_path)
                return (
                    pymupdf_result
                    if pymupdf_result["quality"]["quality_score"] >= docling_result["quality"]["quality_score"]
                    else docling_result
                )
            except PDFIngestionError:
                return pymupdf_result
        except Exception:
            return extract_with_docling(pdf_path)
    raise PDFIngestionError(f"Unknown engine '{engine}'. Choose: pymupdf | docling | auto.")
