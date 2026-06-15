"""
paper_card.py — Dynamic section-aware rule-based paper card builder for Paper2Lab.

Purpose
-------
Builds a clean candidate paper card before LLM/Nemotron refinement.

Design principles
-----------------
- Uses extraction-safe clean_text from pdf_loader.py.
- Uses structured sections and roles instead of raw full-PDF text whenever possible.
- Detects paper_type first, then chooses extraction strategy.
- Avoids ML-only assumptions for systematic reviews, clinical studies, surveys, and reports.
- Keeps references/appendix/boilerplate out of candidate fields.
- Produces llm_evidence_pack for later Modal/Nemotron refinement.

Public API
----------
build_paper_card(extracted: Dict[str, Any]) -> Dict[str, Any]
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

MODEL_PATTERNS = [
    r"\btransformer\b", r"\bself[- ]attention\b", r"\bmulti[- ]head attention\b",
    r"\bscaled dot[- ]product attention\b", r"\bencoder[- ]decoder\b",
    r"\bcnn\b", r"\bconvolutional neural network\b", r"\bu[- ]net\b",
    r"\bbert\b", r"\bgpt\b", r"\bllm\b", r"\blarge language model\b",
    r"\bvision transformer\b", r"\bvit\b", r"\bdiffusion model\b", r"\bgan\b",
    r"\bresnet\b", r"\blstm\b", r"\bgru\b", r"\bsvm\b", r"\brandom forest\b",
    r"\bxgboost\b", r"\blightgbm\b", r"\blogistic regression\b",
    r"\blinear regression\b", r"\bgraph neural network\b", r"\bgnn\b", r"\brag\b",
]

ML_DATASET_PATTERNS = [
    r"\bwmt\s*\d{4}\b", r"\bwsj\b", r"\bwall street journal\b", r"\bpenn treebank\b",
    r"\bcifar[- ]?\d+\b", r"\bimagenet\b", r"\bmnist\b", r"\bcoco\b",
    r"\bglue\b", r"\bsuperglue\b", r"\bsquad\b", r"\bbookscorpus\b",
    r"\bwikipedia\b", r"\bcommonvoice\b", r"\blibrispeech\b",
    r"\b\d+(?:\.\d+)?\s*(?:m|million|b|billion|k|thousand)?\s*"
    r"(?:sentence pairs|sentences|tokens|images|patients|samples|records|documents|cases|examples|instances)\b",
    r"\btraining data\b", r"\btraining dataset\b", r"\bvalidation set\b", r"\btest set\b",
    r"\bdataset\b", r"\bdata source\b", r"\bclinical data\b", r"\bpublic dataset\b", r"\bbenchmark\b",
]

REVIEW_SOURCE_PATTERNS = [
    r"\bpubmed\b", r"\bscopus\b", r"\bweb of knowledge\b", r"\bweb of science\b",
    r"\beric\b", r"\beducational resources and information center\b",
    r"\bcochrane\b", r"\bembase\b", r"\bmedline\b", r"\bgoogle scholar\b",
    r"\bdatabases?\b", r"\barticles?\b", r"\bstudies\b", r"\bpublications?\b",
    r"\brecords identified\b", r"\bselected studies\b", r"\bincluded studies\b",
    r"\bgray literature\b", r"\bgrey literature\b",
]

METHODOLOGY_PATTERNS = [
    # ML / computational methods
    r"\bwe (?:train|trained|fine[- ]?tune|fine[- ]?tuned|evaluate|evaluated|optimize|optimized|pre[- ]?train|pre[- ]?trained)\b",
    r"\bmodel (?:architecture|consists|uses|contains|is trained|was trained)\b",
    r"\barchitecture\b", r"\btraining procedure\b", r"\bexperimental setup\b",
    r"\boptimizer\b", r"\badamw?\b", r"\bsgd\b", r"\blearning rate\b",
    r"\bbatch size\b", r"\bepoch\b", r"\bwarm[- ]?up\b", r"\bscheduler\b",
    r"\bdropout\b", r"\blayer normalization\b", r"\bbatch normalization\b", r"\bweight decay\b",
    r"\btokenization\b", r"\bbyte[- ]pair encoding\b", r"\bpositional encoding\b",
    r"\bself[- ]attention\b", r"\bscaled dot[- ]product attention\b", r"\bmulti[- ]head attention\b",
    r"\bcross[- ]validation\b", r"\btrain[- ]test split\b", r"\brandom seed\b",
    r"\bpre[- ]?processed\b", r"\baugmentation\b",
    # General empirical / review methods
    r"\bsystematic review\b", r"\bliterature review\b", r"\bscoping review\b",
    r"\bdatabases? (?:were|was) searched\b", r"\bsearched\b",
    r"\binclusion criteria\b", r"\bexclusion criteria\b", r"\beligibility criteria\b",
    r"\bscreen(?:ed|ing)\b", r"\bstudies were selected\b", r"\bdata extraction\b",
    r"\bstudy design\b", r"\bparticipants\b", r"\bprocedure\b", r"\bintervention\b",
]

METRIC_PATTERNS = [
    # ML/AI metrics
    r"\bbleu\b", r"\bperplexity\b", r"\baccuracy\b", r"\bprecision\b", r"\brecall\b",
    r"\bf1[- ]?score\b", r"\bf1\b", r"\bauc\b", r"\broc\b", r"\bsensitivity\b",
    r"\bspecificity\b", r"\brmse\b", r"\bmae\b", r"\bmse\b", r"\br\s*[²2]\b",
    r"\bmap\b", r"\biou\b", r"\bwer\b", r"\bcer\b", r"\brouge\b", r"\bbertscore\b",
    r"\bloss\b", r"\bcross[- ]entropy\b",
    # Review / clinical / social-science measurement patterns
    r"\b\d+\s+(?:articles|studies|records|participants|patients|students)\b",
    r"\bfinal review included\b", r"\bwere enrolled\b", r"\bselected for further review\b",
    r"\bbetween\s+(?:january\s+)?\d{4}\s+and\s+(?:january\s+)?\d{4}\b",
    r"\bfrom\s+(?:january\s+)?\d{4}\s+to\s+(?:january\s+)?\d{4}\b",
]

FINDING_PATTERNS = [
    r"\bachieves?\b", r"\boutperforms?\b", r"\bimproves?\b", r"\bincreases?\b",
    r"\bdecreases?\b", r"\bstate[- ]of[- ]the[- ]art\b", r"\bresults show\b",
    r"\bfindings show\b", r"\bsignificantly\b", r"\bsuperior\b", r"\bcomparable\b",
    r"\bconsistently\b", r"\bobtains?\b", r"\bshowed that\b", r"\bfound that\b",
    r"\bpositive (?:responses|attitudes|effects|outcomes)\b", r"\bwas effective\b",
]

CONTRIBUTION_PATTERNS = [
    r"\bwe propose\b", r"\bwe introduce\b", r"\bwe present\b", r"\bwe develop\b",
    r"\bwe designed\b", r"\bwe show\b", r"\bwe demonstrate\b", r"\bwe release\b",
    r"\bwe open[- ]source\b", r"\bthis paper proposes\b", r"\bthis work proposes\b",
    r"\bthis study developed\b", r"\bour contribution\b", r"\bour main contribution\b",
    r"\bnovel\b", r"\bfirst to\b", r"\bfirst systematic review\b",
]

LIMITATION_PATTERNS = [
    r"\blimitation\b", r"\blimitations\b", r"\bfuture work\b", r"\bmore data\b",
    r"\bsmall dataset\b", r"\bfalse[- ]positive\b", r"\bfalse[- ]negative\b",
    r"\bnot sufficient\b", r"\blacking\b", r"\bwe did not\b", r"\bcannot\b",
    r"\bwe leave\b", r"\bdoes not generalize\b", r"\bbias\b", r"\boutside the scope\b",
    r"\bnot evaluated\b", r"\bnot tested\b", r"\black of\b", r"\bmay have led to bias\b",
]

NOISE_MARKERS = [
    "acknowledgement", "acknowledgment", "author contribution", "competing interests",
    "correspondence", "publisher", "open access", "license", "copyright", "gmail.com",
    "references", "bibliography", "arxiv:", "how to cite", "access this article online",
    "quick response code", "website:", "doi:", "www.", "http://", "https://",
    # Known author-contribution noise from some papers
    "llion also experimented", "jakob proposed", "ashish", "noam proposed", "niki selected",
    "aidan designed", "illia", "google brain",
]

AFFILIATION_MARKERS = [
    "department of", "university of", "faculty of", "school of", "institute of",
    "medical sciences", "corresponding author", "email", "journal of education",
]

_FIELD_REJECT: Dict[str, List[str]] = {
    "contributions": ["author", "google brain", "also experimented", "selected work", "department of"],
    "datasets_or_data_sources": [
        "achieves", "outperforms", "state-of-the-art", "results show", "during inference",
        "beam size", "parser training", "section 23", "table 4", "department of",
    ],
    "models_or_methods": ["author", "university", "gmail", "in the following sections", "references"],
    "methodology": ["in the following sections", "to the best of our knowledge", "references"],
    "findings": ["table of contents", "during inference", "parser training", "references"],
}

ML_DATASET_MUST_CONTAIN = [
    "dataset", "data", "wmt", "wsj", "wall street journal", "penn treebank", "bookcorpus",
    "wikipedia", "sentence pairs", "sentences", "tokens", "images", "patients", "samples",
    "records", "cases", "examples", "instances", "benchmark", "training set", "test set",
    "validation set", "english-german", "english-french",
]

REVIEW_SOURCE_MUST_CONTAIN = [
    "pubmed", "scopus", "web of knowledge", "web of science",
    "educational resources", "cochrane", "embase", "medline",
    "google scholar", "database", "databases", "studies",
    "articles", "publications", "records", "gray literature", "grey literature",
]

# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _strip_doi_noise(text: str) -> str:
    text = text or ""
    text = re.sub(r"\bdoi\s*[:：]?\s*10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", " ", text)
    return text


def _clean(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = _strip_doi_noise(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean(text).lower()).strip()


def _bad_sentence_quality(sentence: str) -> bool:
    """Reject merged-column, affiliation, citation, and boilerplate artifacts."""
    s = _clean(sentence)
    low = s.lower()

    if not s:
        return True

    if any(x in low for x in AFFILIATION_MARKERS):
        return True

    # Too many citation markers often means merged reference/body text.
    if len(re.findall(r"\[\d+\]", s)) >= 2:
        return True

    # Known symptoms of two-column stitching / line interleaving.
    bad_regexes = [
        r"\bneed\s+this systematic review\b",
        r"\bof the there\b",
        r"\band the that\b",
        r"\bwere as follows: being\b",
        r"\bdata were included to improve\b",
        r"\bto adopt a new style of learning\s*\.\s*medical courses\b",
        r"\bfisher \(\d{4}\) discuss intended studies\b",
    ]
    if any(re.search(p, low) for p in bad_regexes):
        return True

    # Merged sentences are usually very long and contain unrelated cues.
    if len(s.split()) > 55 and any(x in low for x in [
        "students are required", "there is a good deal", "department", "university",
        "corresponding author", "access this article",
    ]):
        return True

    # Odd punctuation/table artifacts.
    if s.count("|") >= 2 or s.count("%") >= 6 or s.count("@") >= 1:
        return True

    return False


def _is_noise(sentence: str) -> bool:
    low = sentence.lower()
    if any(m in low for m in NOISE_MARKERS):
        return True
    if len(sentence.split()) > 85:
        return True
    if re.search(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z ]{2,50}$", sentence):
        return True
    if _bad_sentence_quality(sentence):
        return True
    return False


def _split_sentences(text: str) -> List[str]:
    text = _clean(text)
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences: List[str] = []
    for s in raw:
        s = _clean(s)
        if 35 <= len(s) <= 420 and not _is_noise(s):
            sentences.append(s)
    return sentences


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        item = _clean(item)
        key = _normalize_key(item)[:220]
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _section_title(sec: Dict[str, Any]) -> str:
    return _clean(sec.get("title") or "")


def _section_role(sec: Dict[str, Any]) -> str:
    return sec.get("role") or "other"


def _sections_by_role(
    extracted: Dict[str, Any],
    roles: List[str],
    title_contains: List[str] | None = None,
) -> str:
    chunks: List[str] = []
    title_contains = [t.lower() for t in (title_contains or [])]
    paper_title = _clean(extracted.get("title") or "").lower()

    blocked_roles = {"references", "appendix", "boilerplate"}
    blocked_titles = {"front matter", "keywords", "keywords:", "table of contents"}

    for sec in extracted.get("sections", []):
        title_raw = _section_title(sec)
        title = title_raw.lower()
        role = _section_role(sec)

        if role in blocked_roles:
            continue
        if title in blocked_titles:
            continue
        if paper_title and title == paper_title:
            continue

        if role in roles or any(t in title for t in title_contains):
            chunks.append(sec.get("text", ""))

    return "\n".join(chunks)


def _find_sentences(
    text: str,
    patterns: List[str],
    max_items: int = 8,
    require_number: bool = False,
) -> List[str]:
    found: List[str] = []
    for sentence in _split_sentences(text):
        if require_number and not re.search(r"\d", sentence):
            continue
        if any(re.search(p, sentence, flags=re.IGNORECASE) for p in patterns):
            found.append(sentence)
        if len(found) >= max_items:
            break
    return _dedupe(found)


def _filter_field(items: List[str], field: str, paper_type: str = "general_research") -> List[str]:
    reject_terms = _FIELD_REJECT.get(field, [])
    filtered: List[str] = []

    for item in items:
        item = _clean(item)
        low = item.lower()
        if not item or _is_noise(item):
            continue
        if any(term in low for term in reject_terms):
            continue

        if field == "datasets_or_data_sources":
            if paper_type == "systematic_review":
                if not any(term in low for term in REVIEW_SOURCE_MUST_CONTAIN):
                    continue
            else:
                if not any(term in low for term in ML_DATASET_MUST_CONTAIN):
                    continue
            if any(bad in low for bad in ["beam size", "during inference", "parser training", "section 23"]):
                continue

        filtered.append(item)

    return _dedupe(filtered)


# ---------------------------------------------------------------------------
# Dynamic paper type and field inference
# ---------------------------------------------------------------------------

def _top_terms(text: str, k: int = 10) -> List[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "has",
        "have", "had", "their", "they", "into", "using", "used", "study", "paper", "article",
        "results", "method", "methods", "data", "based", "between", "through", "there",
    }
    words = re.findall(r"\b[a-z][a-z]{3,}\b", text.lower())
    words = [w for w in words if w not in stop]
    return [w for w, _ in Counter(words).most_common(k)]


def _infer_paper_type(title: str, abstract: str, sections: List[Dict[str, Any]], clean_text: str) -> str:
    title_abs = f"{title} {abstract}".lower()
    section_titles = " ".join(_section_title(s) for s in sections).lower()
    probe = f"{title_abs} {section_titles} {clean_text[:9000].lower()}"

    review_score = 0
    for term in [
        "systematic review",
        "systematic literature search",
        "systematic literature review",
        "literature search",
        "literature review",
        "prisma",
        "study selection",
        "bibliographic search",
        "databases were searched",
        "database search",
        "included studies",
        "included articles",
        "screening",
        "eligibility criteria",
    ]:
        if term in probe:
            review_score += 1

    if review_score >= 2:
        return "systematic_review"

    if any(x in probe for x in ["meta-analysis", "meta analysis", "scoping review"]):
        return "systematic_review"
    
    narrative_review_terms = [
        "review of important findings",
        "review of previous research",
        "previous research on",
        "literature on",
        "research on teacher education",
        "implications for improving",
    ]
    if any(term in probe for term in narrative_review_terms):
        return "survey_or_review"

    if any(x in probe for x in ["randomized controlled trial", "cohort", "case-control", "clinical trial"]):
        return "clinical_study"

    if any(x in probe for x in ["transformer", "bert", "neural network", "optimizer", "training loss", "fine-tuned", "imagenet", "bleu"]):
        return "machine_learning"

    if any(x in probe for x in ["survey", "questionnaire", "respondents", "participants"]):
        return "survey_study"

    if any(x in probe for x in ["recommendations", "checklist", "best practices"]):
        return "guide_or_report"

    return "general_research"


def _infer_field(title: str, abstract: str, clean_text: str, paper_type: str) -> str:
    title_abs = f"{title} {abstract}".lower()
    probe = f"{title_abs} {clean_text[:5000].lower()}"

    domain_scores: Dict[str, int] = {
        "Education": 0,
        "Natural Language Processing": 0,
        "Computer Vision": 0,
        "Medical / Clinical Research": 0,
        "Medical AI": 0,
        "Biology / Life Sciences": 0,
        "Graph Learning": 0,
        "Reinforcement Learning": 0,
        "Generative Models": 0,
        "Social Science": 0,
        "Machine Learning": 0,
    }

    weighted_terms: Dict[str, List[str]] = {
        "Education": ["education", "educational", "academic performance", "students", "higher education", "preclinical academic"],
        "Natural Language Processing": ["translation", "language model", "question answering", "summarization", "bleu", "token", "bert", "gpt", "corpus"],
        "Computer Vision": ["image classification", "object detection", "segmentation", "imagenet", "resnet", "vision transformer", "pixel"],
        "Medical / Clinical Research": ["patient", "clinical", "cohort", "diagnosis", "mortality", "disease", "treatment", "medical students"],
        "Medical AI": ["clinical prediction", "medical image", "radiograph", "x-ray", "ct scan", "mri", "ehr", "icu", "machine learning"],
        "Biology / Life Sciences": ["protein", "molecule", "drug", "genomics", "dna", "rna", "crispr", "gene"],
        "Graph Learning": ["graph neural network", "gnn", "node classification", "link prediction", "message passing"],
        "Reinforcement Learning": ["reward", "agent", "policy", "q-learning", "ppo", "dqn", "actor-critic"],
        "Generative Models": ["diffusion model", "gan", "generative adversarial", "vae", "denoising"],
        "Social Science": ["survey", "questionnaire", "social science", "respondents", "interviews"],
        "Machine Learning": ["machine learning", "deep learning", "neural network", "training", "optimizer", "classification"],
    }

    for domain, terms in weighted_terms.items():
        for term in terms:
            if term in probe:
                domain_scores[domain] += 1
            if term in title_abs:
                domain_scores[domain] += 2

    # Paper-type-aware tie breaks.
    if paper_type == "systematic_review":
        if domain_scores["Education"] >= 2:
            return "Education"
        if domain_scores["Medical / Clinical Research"] >= 2:
            return "Medical / Clinical Research"
        if domain_scores["Social Science"] >= 2:
            return "Social Science"

    if domain_scores["Medical AI"] >= 3 and domain_scores["Machine Learning"] >= 2:
        return "Medical AI"

    best = max(domain_scores, key=domain_scores.get)
    if domain_scores[best] <= 1:
        return "General Research"
    return best


# ---------------------------------------------------------------------------
# Field-specific extraction helpers
# ---------------------------------------------------------------------------

def _extract_research_question(abstract: str, intro: str, title: str, paper_type: str) -> str:
    text = _clean(f"{abstract} {intro}")

    if paper_type == "systematic_review":
        review_markers = [
            "purpose", "aim", "objective", "intended", "present review", "systematic review",
            "with the purpose", "the aim of", "the objective of",
        ]
        for sentence in _split_sentences(text):
            low = sentence.lower()
            if any(m in low for m in review_markers):
                return sentence

    markers = [
        "we propose", "we introduce", "we investigate", "we evaluate", "we present",
        "we developed", "this paper proposes", "this work proposes", "the goal",
        "the aim", "the objective", "our goal", "our aim", "this study aims",
    ]
    for sentence in _split_sentences(text):
        low = sentence.lower()
        if any(m in low for m in markers):
            return sentence

    clean_abstract = _split_sentences(abstract)
    if clean_abstract:
        return clean_abstract[0]

    return title or ""


def _extract_contributions(abstract: str, intro: str, conclusion: str, paper_type: str) -> List[str]:
    text = f"{abstract} {intro} {conclusion}"
    items = _find_sentences(text, CONTRIBUTION_PATTERNS, 8)
    if paper_type == "systematic_review" and not items:
        # Many reviews do not have explicit contribution language. Avoid hallucinating.
        return []
    return _filter_field(items, "contributions", paper_type)


def _extract_methodology(methods: str, experiments: str, intro: str, paper_type: str) -> List[str]:
    primary = f"{methods}\n{experiments}"

    if paper_type == "systematic_review":
        review_patterns = [
            r"\b(?:pubmed|scopus|web of knowledge|web of science|eric|cochrane|embase|medline)\b.*\bsearched\b",
            r"\bdatabases?\b.*\bsearched\b",
            r"\binclusion criteria\b.*",
            r"\bexclusion criteria\b.*",
            r"\beligibility criteria\b.*",
            r"\btitles and abstracts were screened\b.*",
            r"\barticles were imported\b.*",
            r"\bdata extraction\b.*",
            r"\bsystematic review was conducted\b.*",
        ]
        items = _find_sentences(primary, review_patterns, max_items=10)
        if len(items) < 3:
            items += _find_sentences(f"{intro}\n{primary}", review_patterns, max_items=10)
        return _filter_field(items, "methodology", paper_type)[:8]

    items = _find_sentences(primary, METHODOLOGY_PATTERNS, max_items=12)
    if len(items) < 3:
        items += _find_sentences(intro, METHODOLOGY_PATTERNS, max_items=5)
    return _filter_field(items, "methodology", paper_type)[:10]


def _extract_datasets(clean_text: str, methods: str, experiments: str, results: str, paper_type: str) -> List[str]:
    priority_text = f"{methods}\n{experiments}\n{results}"

    if paper_type == "systematic_review":
        items = _find_sentences(priority_text, REVIEW_SOURCE_PATTERNS, max_items=14)
        if len(items) < 4:
            items += _find_sentences(clean_text[:12000], REVIEW_SOURCE_PATTERNS, max_items=14)

        # Also add compact database names when explicitly found.
        compact: List[str] = []
        db_names = [
            "PubMed", "Scopus", "Web of Knowledge", "Web of Science", "ERIC",
            "Educational Resources and Information Center", "Cochrane", "Embase", "Medline", "Google Scholar",
        ]
        low = priority_text.lower() + " " + clean_text[:8000].lower()
        for name in db_names:
            if name.lower() in low:
                compact.append(name)
        compact += _filter_field(items, "datasets_or_data_sources", paper_type)
        return _dedupe(compact)[:10]

    items = _find_sentences(priority_text, ML_DATASET_PATTERNS, 12)
    if len(items) < 4:
        items += _find_sentences(clean_text[:15000], ML_DATASET_PATTERNS, 12)
    return _filter_field(items, "datasets_or_data_sources", paper_type)[:10]


def _extract_models(clean_text: str, methods: str, experiments: str, paper_type: str) -> List[str]:
    if paper_type in {"systematic_review", "guide_or_report", "survey_study"}:
        return []
    priority_text = f"{methods}\n{experiments}"
    items = _find_sentences(priority_text, MODEL_PATTERNS, 12)
    if len(items) < 4:
        items += _find_sentences(clean_text[:15000], MODEL_PATTERNS, 12)
    return _filter_field(items, "models_or_methods", paper_type)[:10]


def _extract_metrics(clean_text: str, results: str, experiments: str, methods: str, paper_type: str) -> List[str]:
    priority_text = f"{results}\n{experiments}\n{methods}"

    if paper_type == "systematic_review":
        review_metric_patterns = [
            r"\b\d+\s+(?:articles|studies|records|publications)\b.*",
            r"\b(?:final review|review) included\s+\d+\b.*",
            r"\b\d+\s+articles were selected\b.*",
            r"\bbetween\s+(?:january\s+)?\d{4}\s+and\s+(?:january\s+)?\d{4}\b.*",
            r"\bfrom\s+(?:january\s+)?\d{4}\s+to\s+(?:january\s+)?\d{4}\b.*",
        ]
        items = _find_sentences(priority_text, review_metric_patterns, 10, require_number=True)
        if len(items) < 3:
            items += _find_sentences(clean_text[:12000], review_metric_patterns, 10, require_number=True)
        return _dedupe(items)[:8]

    items = _find_sentences(priority_text, METRIC_PATTERNS, 12, require_number=True)
    if len(items) < 3:
        items += _find_sentences(clean_text[:15000], METRIC_PATTERNS, 10, require_number=True)
    if not items:
        items = _find_sentences(priority_text or clean_text[:15000], METRIC_PATTERNS, 8)
    return _dedupe([x for x in items if "references" not in x.lower()])[:10]


def _extract_findings(results: str, conclusion: str, abstract: str, paper_type: str) -> List[str]:
    text = f"{results} {conclusion} {abstract}"
    items = _find_sentences(text, FINDING_PATTERNS, 10)
    return _filter_field(items, "findings", paper_type)[:5]


def _extract_limitations(clean_text: str, limitations: str, discussion: str, conclusion: str, paper_type: str) -> List[str]:
    text = f"{limitations} {discussion} {conclusion} {clean_text[-6000:]}"
    return _filter_field(_find_sentences(text, LIMITATION_PATTERNS, 8), "limitations", paper_type)[:6]


def _missing_repro_info(card: Dict[str, Any], extracted: Dict[str, Any], paper_type: str) -> List[str]:
    missing: List[str] = []
    method_text = _sections_by_role(extracted, ["methodology", "experiments"])
    low = method_text.lower()

    if paper_type == "systematic_review":
        checks = {
            "search date range is not clearly specified": ["from", "between", "1987", "2018", "date", "time limitation"],
            "inclusion/exclusion criteria are not clearly specified": ["inclusion criteria", "exclusion criteria", "eligibility criteria"],
            "screening process is not clearly specified": ["screened", "titles and abstracts", "two independent"],
            "quality assessment method is not clearly specified": ["quality", "assessment", "best evidence", "risk of bias"],
        }
        for label, terms in checks.items():
            if not any(t in low for t in terms):
                missing.append(label)
        if not card.get("datasets_or_data_sources"):
            missing.append("bibliographic databases or study sources could not be reliably extracted")
        return missing[:6]

    if paper_type == "machine_learning":
        checks = {
            "training hyperparameters are incomplete": ["learning rate", "batch size", "epoch", "steps"],
            "random seed is not specified": ["seed", "random seed"],
            "code availability is not clearly specified": ["code", "github", "repository"],
            "dataset split details are not clearly specified": ["train", "validation", "test", "split"],
        }
    else:
        checks = {
            "study design details are incomplete": ["study design", "method", "procedure"],
            "sample/data source details are incomplete": ["participants", "patients", "samples", "data source", "dataset"],
            "analysis or measurement method is not clearly specified": ["analysis", "measure", "outcome", "metric"],
        }

    for label, terms in checks.items():
        if not any(t in low for t in terms):
            missing.append(label)
    if not card.get("datasets_or_data_sources") and paper_type == "machine_learning":
        missing.append("dataset or data source could not be reliably extracted")
    return missing[:6]


# ---------------------------------------------------------------------------
# Evidence pack
# ---------------------------------------------------------------------------

def _llm_evidence_pack(extracted: Dict[str, Any], paper_card: Dict[str, Any]) -> Dict[str, Any]:
    compact_sections: List[Dict[str, Any]] = []
    for sec in extracted.get("sections", []):
        text = _clean(sec.get("text", ""))
        if not text:
            continue
        compact_sections.append({
            "title": sec.get("title", ""),
            "role_hint": sec.get("role", "other"),
            "page_start": sec.get("page_start"),
            "page_end": sec.get("page_end"),
            "preview": text[:2000],
        })

    candidate = {k: v for k, v in paper_card.items() if k != "llm_evidence_pack"}
    return {
        "system_prompt": (
            "You are a scientific paper analyst. Correct the candidate JSON paper card using only the provided evidence. "
            "Do not invent facts. Remove boilerplate, references, author affiliations, acknowledgements, duplicate claims, "
            "and generic text. Return only valid JSON with the same keys as candidate_paper_card."
        ),
        "instruction": (
            "Refine candidate_paper_card. Keep facts specific, concise, and grounded in section previews, captions, and tables. "
            "Respect paper_type: for systematic reviews, extract databases, inclusion/exclusion criteria, screening process, "
            "selected studies, synthesis method, findings, and review limitations instead of ML hyperparameters. "
            "Use null or [] when a field cannot be determined."
        ),
        "candidate_paper_card": candidate,
        "section_previews": compact_sections[:16],
        "captions": extracted.get("captions", [])[:10],
        "tables": [_json_safe(t) for t in extracted.get("tables", [])[:4]],
        "metadata": {
            "references_count": len(extracted.get("references", [])),
            "references_removed_from_body": bool(extracted.get("references_text")),
            "appendix_removed_from_body": bool(extracted.get("appendix_text")),
            "quality": extracted.get("quality", {}),
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_paper_card(extracted: Dict[str, Any]) -> Dict[str, Any]:
    title = extracted.get("title") or ""
    abstract = extracted.get("abstract") or ""
    clean_text = extracted.get("clean_text") or extracted.get("text") or ""
    sections = extracted.get("sections", [])

    intro = _sections_by_role(extracted, ["introduction", "background"])
    methods = _sections_by_role(extracted, ["methodology"])
    experiments = _sections_by_role(extracted, ["experiments"])
    results = _sections_by_role(extracted, ["results", "experiments"])
    discussion = _sections_by_role(extracted, ["discussion"])
    limitations_text = _sections_by_role(extracted, ["limitations"])
    conclusion = _sections_by_role(extracted, ["conclusion"])

    paper_type = _infer_paper_type(title, abstract, sections, clean_text)
    field = _infer_field(title, abstract, clean_text, paper_type)

    card: Dict[str, Any] = {
        "title": title or None,
        "field": field,
        "paper_type": paper_type,
        "research_question": _extract_research_question(abstract, intro, title, paper_type) or None,
        "contributions": _extract_contributions(abstract, intro, conclusion, paper_type),
        "methodology": _extract_methodology(methods, experiments, intro, paper_type),
        "datasets_or_data_sources": _extract_datasets(clean_text, methods, experiments, results, paper_type),
        "models_or_methods": _extract_models(clean_text, methods, experiments, paper_type),
        "metrics_or_measurements": _extract_metrics(clean_text, results, experiments, methods, paper_type),
        "key_findings": _extract_findings(results, conclusion, abstract, paper_type),
        "limitations": _extract_limitations(clean_text, limitations_text, discussion, conclusion, paper_type),
        "missing_reproducibility_info": [],
        "metadata": {
            "source_pdf": extracted.get("source_pdf"),
            "num_pages": extracted.get("num_pages"),
            "extraction_engine": extracted.get("extraction_engine"),
            "quality": extracted.get("quality", {}),
            "references_count": len(extracted.get("references", [])),
            "references_removed_from_body": bool(extracted.get("references_text")),
            "appendix_removed_from_body": bool(extracted.get("appendix_text")),
            "section_roles": extracted.get("quality", {}).get("section_roles", []),
            "top_terms": _top_terms(f"{title} {abstract} {clean_text[:5000]}", 10),
        },
        "source_pdf": extracted.get("source_pdf"),
        "annotation_version": "v1.0",
    }

    card["missing_reproducibility_info"] = _missing_repro_info(card, extracted, paper_type)
    card["llm_evidence_pack"] = _llm_evidence_pack(extracted, card)
    return card
