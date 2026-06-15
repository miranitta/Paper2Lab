"""
schema.py — Paper2Lab data contracts.

Field-agnostic contracts for PDF extraction across ML, NLP, CV, biomedical,
physics, education, social-science, economics, and interdisciplinary papers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


KNOWN_ROLES = frozenset({
    "front_matter",
    "abstract",
    "keywords",
    "introduction",
    "related_work",
    "background",
    "theory",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "future_work",
    "conclusion",
    "references",
    "appendix",
    "boilerplate",
    "other",
})


@dataclass
class Section:
    title: str
    text: str
    level: int = 1
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    role: str = "other"
    word_count: int = 0

    def __post_init__(self) -> None:
        if self.role not in KNOWN_ROLES:
            self.role = "other"
        if not self.word_count:
            self.word_count = len((self.text or "").split())


@dataclass
class Caption:
    label: str
    caption: str
    page_number: Optional[int] = None


@dataclass
class Table:
    page_number: Optional[int]
    table_index: int
    data: Any
    engine: str = "unknown"
    caption: Optional[str] = None


@dataclass
class DocumentExtraction:
    source_pdf: str
    title: Optional[str]
    abstract: Optional[str]
    text: str
    clean_text: str
    raw_text: str = ""
    num_pages: Optional[int] = None
    sections: List[Section] = field(default_factory=list)
    all_sections: List[Section] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    references_text: str = ""
    appendix_text: str = ""
    boilerplate_text: str = ""
    captions: List[Caption] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    pages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    extraction_engine: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
