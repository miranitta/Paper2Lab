"""
visual_explainer.py — Caption/table understanding for Paper2Lab.

This is caption-grounded visual understanding. It does not use image pixels yet.
For the hackathon MVP, this gives useful figure/table summaries without GPU cost.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


_METRIC_WORDS = [
    "accuracy", "precision", "recall", "f1", "auc", "bleu", "rouge", "loss", "perplexity",
    "score", "performance", "results", "comparison", "evaluation", "training cost", "p-value",
]

_METHOD_WORDS = [
    "architecture", "pipeline", "framework", "workflow", "model", "method", "procedure",
    "attention", "encoder", "decoder", "algorithm", "overview",
]

_DATA_WORDS = [
    "dataset", "data", "samples", "patients", "images", "sentences", "articles", "studies",
    "distribution", "statistics", "characteristics",
]


def _clean(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:\n\t")


def _label_type(label: str) -> str:
    low = (label or "").lower()
    if "table" in low or "tbl" in low:
        return "table"
    if "figure" in low or "fig" in low:
        return "figure"
    if "algorithm" in low:
        return "algorithm"
    if "scheme" in low:
        return "scheme"
    return "visual"


def _purpose(caption: str, visual_type: str) -> str:
    low = caption.lower()
    if any(w in low for w in _METHOD_WORDS):
        return "method_or_architecture"
    if any(w in low for w in _METRIC_WORDS):
        return "results_or_evaluation"
    if any(w in low for w in _DATA_WORDS):
        return "data_or_dataset_description"
    if visual_type == "table":
        return "structured_results_or_metadata"
    return "illustrative_figure"


def _summary_from_caption(label: str, caption: str, visual_type: str) -> str:
    caption = _clean(caption)
    if not caption:
        return f"{label} is a {visual_type}, but no caption text was extracted."
    # Keep concise, but grounded in caption text.
    if len(caption) <= 220:
        return caption
    first_sentence = re.split(r"(?<=[.!?])\s+", caption)[0]
    return _clean(first_sentence[:260])


def _summarize_table_data(table: Dict[str, Any]) -> str | None:
    data = table.get("data")
    if not isinstance(data, list) or not data:
        return None
    rows = [r for r in data if isinstance(r, list)]
    if not rows:
        return None
    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=0)
    header = rows[0] if rows else []
    header_text = ", ".join(str(x).strip() for x in header if str(x).strip())[:180]
    if header_text:
        return f"Extracted table with approximately {n_rows} rows and {n_cols} columns. Header fields include: {header_text}."
    return f"Extracted table with approximately {n_rows} rows and {n_cols} columns."


def explain_figures_and_tables(extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return concise explanations for extracted captions and tables."""
    outputs: List[Dict[str, Any]] = []

    for cap in extracted.get("captions", []) or []:
        label = cap.get("label", "")
        caption = _clean(cap.get("caption", ""))
        visual_type = _label_type(label)
        outputs.append({
            "label": label,
            "type": visual_type,
            "purpose": _purpose(caption, visual_type),
            "summary": _summary_from_caption(label, caption, visual_type),
            "evidence": caption,
            "page_number": cap.get("page_number"),
        })

    # Add tables that have data but no caption match.
    existing_table_pages = {(o.get("page_number"), o.get("label")) for o in outputs if o.get("type") == "table"}
    for table in extracted.get("tables", []) or []:
        page = table.get("page_number")
        caption = _clean(table.get("caption") or "")
        label = f"Table extracted on page {page}" if page is not None else "Extracted table"
        if (page, label) in existing_table_pages:
            continue
        data_summary = _summarize_table_data(table)
        outputs.append({
            "label": label,
            "type": "table",
            "purpose": _purpose(caption or data_summary or "", "table"),
            "summary": caption or data_summary or "A table was detected, but its content could not be summarized reliably.",
            "evidence": caption or data_summary or "",
            "page_number": page,
        })

    return outputs[:20]
