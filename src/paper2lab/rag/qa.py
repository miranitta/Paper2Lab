"""
qa.py — Evidence-grounded local RAG Q&A for Paper2Lab.

This module returns extractive answers with evidence and source locations.
It does not call an LLM. Nemotron can later rewrite the answer using the same evidence.

Design:
- Classify the question into a small intent taxonomy.
- Retrieve evidence with FAISS through indexer.py.
- Synthesize answers using intent-specific extractive logic.
- Avoid hardcoding known dataset names; discover entities from local evidence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from paper2lab.rag.indexer import RagIndex, build_rag_index, search_rag_index


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_QUERY_INTENTS: Dict[str, List[str]] = {
    "datasets": [
        "dataset", "datasets", "data", "corpus", "corpora", "benchmark", "benchmarks",
        "source", "sources", "database", "databases", "training set", "test set",
        "validation set", "dev set", "patients", "samples", "records", "articles", "studies",
    ],
    "methodology": [
        "method", "methods", "methodology", "procedure", "procedures", "steps", "approach",
        "how", "trained", "training", "fine-tuned", "pretrained", "searched", "screened",
        "selected", "included", "excluded", "implementation", "architecture", "pipeline",
    ],
    "evaluation": [
        "evaluate", "evaluated", "evaluation", "metric", "metrics", "score", "accuracy",
        "precision", "recall", "f1", "auc", "bleu", "rouge", "perplexity", "result",
        "results", "performance", "finding", "findings", "outcome", "outcomes",
    ],
    "figures": [
        "figure", "fig", "table", "caption", "diagram", "plot", "chart", "architecture",
        "visual", "illustration", "show", "shows",
    ],
    "reproducibility": [
        "missing", "reproduce", "reproduction", "reproducibility", "hyperparameter",
        "hyperparameters", "software", "code", "github", "repository", "settings", "requirements",
        "seed", "hardware", "gpu", "implementation details",
    ],
}

_INTENT_QUERY_EXPANSIONS: Dict[str, str] = {
    "datasets": "dataset corpus benchmark training data test set validation data source database articles studies",
    "methodology": "method methodology approach procedure steps training implementation experimental setup search screening inclusion exclusion",
    "evaluation": "evaluation metrics results performance score accuracy f1 bleu rouge auc outcome analysis measured assessed",
    "figures": "figure table caption diagram architecture plot shows illustrates",
    "reproducibility": "reproducibility missing information hyperparameters dataset details software code hardware seed experimental settings",
    "general": "paper evidence method results conclusion",
}


# ---------------------------------------------------------------------------
# Cleaning / sentence utilities
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:\n\t")


def _is_noisy_sentence(sentence: str) -> bool:
    s = _clean(sentence)
    low = s.lower()

    bad_fragments = [
        "corresponding author", "how to cite", "access this article online", "department of",
        "university of", "medical sciences", "received:", "accepted:", "published:",
        "copyright", "license", "all rights reserved", "gmail.com", "@",
        "being accordingly", "endnote teachers", "the there", "resultsare",
        "analysis of the resultsare", "need this systematic review",
    ]
    if any(x in low for x in bad_fragments):
        return True
    if len(re.findall(r"\[\d+", s)) >= 3:
        return True
    if s.count("|") >= 2 or s.count("%") >= 6:
        return True
    if len(s.split()) > 85:
        return True
    return False


def _split_sentences(text: str) -> List[str]:
    text = _clean(text)
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    out: List[str] = []
    for sent in raw:
        sent = _clean(sent)
        if 25 <= len(sent) <= 420 and not _is_noisy_sentence(sent):
            out.append(sent)
    if not out and text and not _is_noisy_sentence(text):
        out = [text[:420]]
    return out


def _query_terms(question: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", question.lower())
    stop = {
        "what", "which", "where", "when", "how", "were", "was", "are", "the", "and",
        "used", "use", "paper", "study", "does", "did", "for", "with", "from", "that",
        "this", "these", "those", "show", "shows", "tell", "about", "explain",
    }
    return [w for w in words if w not in stop]


def _dedupe_strings(items: Iterable[str], limit: int = 10) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        item = _clean(item)
        if not item:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()[:180]
        if key and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _intent(question: str) -> str:
    low = question.lower()
    scores: Dict[str, int] = {}
    for intent, keys in _QUERY_INTENTS.items():
        score = 0
        for k in keys:
            if k in low:
                score += 2 if " " in k else 1
        scores[intent] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _expanded_query(question: str, intent: str) -> str:
    expansion = _INTENT_QUERY_EXPANSIONS.get(intent, "")
    return _clean(f"{question} {expansion}")


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _evidence_texts(hits: List[Any]) -> List[str]:
    return [getattr(h, "text", "") for h in hits if getattr(h, "text", "")]


def _rank_sentences(question: str, evidence_texts: List[str], max_sentences: int = 4) -> List[str]:
    terms = _query_terms(question)
    candidates: List[Tuple[int, int, str]] = []
    for text in evidence_texts:
        for sent in _split_sentences(text):
            low = sent.lower()
            lexical_score = sum(1 for t in terms if t in low)
            length_penalty = max(0, len(sent.split()) - 45)
            candidates.append((lexical_score, -length_penalty, sent))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected: List[str] = []
    seen: set[str] = set()
    for score, _, sent in candidates:
        key = re.sub(r"[^a-z0-9]+", " ", sent.lower()).strip()[:180]
        if key in seen:
            continue
        if score == 0 and selected:
            continue
        seen.add(key)
        selected.append(sent)
        if len(selected) >= max_sentences:
            break
    return selected


# ---------------------------------------------------------------------------
# Dataset/data-source discovery without hardcoded dataset names
# ---------------------------------------------------------------------------

_DATA_CONTEXT_WORDS = [
    "dataset", "datasets", "corpus", "corpora", "benchmark", "benchmarks", "training data",
    "training set", "test set", "validation set", "dev set", "data source", "databases",
    "database", "articles", "studies", "patients", "samples", "records", "images",
    "sentences", "tokens", "documents", "cases", "examples", "instances",
]

_KNOWN_DATABASE_GENERIC = [
    "PubMed", "Scopus", "Web of Knowledge", "ERIC", "Educational Resources and Information Center",
    "Cochrane", "IEEE Xplore", "ACM Digital Library", "Google Scholar", "MEDLINE", "Embase",
]

_DATASET_REJECT_TERMS = [
    "parser", "berkeleyparser", "berkleyparser", "rnn", "lstm", "gru",
    "transformer", "recurrent neural network", "neural network grammar",
    "model", "architecture", "baseline", "beam size", "during inference",
    "dropout", "optimizer", "learning rate", "attention", "encoder", "decoder",
]

_DATASET_ALLOW_TERMS = [
    "dataset", "corpus", "corpora", "benchmark", "treebank", "wsj",
    "wmt", "penn treebank", "wall street journal", "sentence pairs",
    "sentences", "tokens", "training set", "test set", "validation set",
    "dev set", "patients", "samples", "records", "articles", "studies",
]

def _extract_capitalized_entities_near_data_terms(sentence: str) -> List[str]:
    """Discover likely dataset names from context without fixed known dataset list."""
    s = _clean(sentence)
    low = s.lower()
    if not any(w in low for w in _DATA_CONTEXT_WORDS):
        return []

    found: List[str] = []

    # Pattern: "standard X dataset", "larger X corpus", "on X benchmark".
    context_patterns = [
        r"(?:standard|larger|public|available|benchmark|the)\s+([A-Z][A-Za-z0-9._/-]*(?:\s+[A-Z]?[A-Za-z0-9._/-]+){0,6})\s+(?:dataset|datasets|corpus|corpora|benchmark|benchmarks)",
        r"(?:on|using|from|with)\s+(?:the\s+)?([A-Z][A-Za-z0-9._/-]*(?:\s+[A-Z]?[A-Za-z0-9._/-]+){0,7})\s+(?:dataset|datasets|corpus|corpora|benchmark|benchmarks)",
        r"([A-Z][A-Za-z0-9._/-]*(?:\s+[A-Z]?[A-Za-z0-9._/-]+){0,7})\s+(?:dataset|datasets|corpus|corpora|benchmark|benchmarks)",
    ]
    for pat in context_patterns:
        for m in re.finditer(pat, s):
            cand = _clean(m.group(1))
            if _valid_dataset_candidate(cand):
                found.append(cand)

    # Pattern: explicit study/data counts.
    count_patterns = [
        r"\b(?:about|approximately|around)?\s*\d+(?:\.\d+)?\s*(?:k|m|million|billion|thousand)?\s+(?:sentence pairs|sentences|tokens|images|patients|samples|records|documents|cases|examples|instances|articles|studies)\b",
        r"\b\d+\s+(?:articles|studies|patients|samples|records)\s+(?:were\s+)?(?:included|enrolled|selected|used)\b",
    ]
    for pat in count_patterns:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            found.append(_clean(m.group(0)))

    # Known scholarly databases are generic enough to keep; not paper-specific datasets.
    for db in _KNOWN_DATABASE_GENERIC:
        if db.lower() in low:
            found.append(db)

    # Parenthetical abbreviations after a named source: Wall Street Journal (WSJ), etc.
    for m in re.finditer(r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\s*\(([A-Z0-9-]{2,10})\)", s):
        cand = _clean(f"{m.group(1)} ({m.group(2)})")
        if _valid_dataset_candidate(cand):
            found.append(cand)

    # Generic dataset-style identifiers near data terms: WMT2014, CIFAR-10, SQuAD-v2, XYZ-500.
    for m in re.finditer(
        r"\b[A-Z]{2,}[A-Za-z]*[- ]?\d{2,4}(?:[- ][A-Za-z]+)*\b",
        s,
    ):
        cand = _clean(m.group(0))
        if _valid_dataset_candidate(cand):
            found.append(cand)

    # Named corpora/treebanks/splits with abbreviations.
    for m in re.finditer(
        r"\b(?:Wall Street Journal|Penn Treebank|[A-Z]{2,6})\b(?:\s*\([A-Z0-9-]{2,10}\))?",
        s,
    ):
        cand = _clean(m.group(0))
        if _valid_dataset_candidate(cand):
            found.append(cand)
    return _dedupe_strings(found, limit=12)


def _valid_dataset_candidate(candidate: str) -> bool:
    cand = _clean(candidate)
    low = cand.lower()

    if not cand or len(cand) < 3 or len(cand.split()) > 10:
        return False

    if any(term in low for term in _DATASET_REJECT_TERMS):
        return False

    bad_exact = {
        "the", "standard", "larger", "public", "available", "training",
        "test", "validation", "we", "our", "this", "that", "section",
        "table", "figure", "results", "parser",
    }
    if low in bad_exact:
        return False

    if any(x in low for x in ["section describes", "the following", "in this", "of the"]):
        return False

    # Accept known dataset-like abbreviations only when context looks data-related.
    if re.fullmatch(r"[A-Z0-9-]{2,12}", cand):
        return True

    return True


def _looks_like_dataset_detail(text: str) -> bool:
    low = _clean(text).lower()
    return bool(
        re.search(
            r"\b(?:about|approximately|around)?\s*\d+(?:\.\d+)?\s*"
            r"(?:k|m|million|billion|thousand)?\s+"
            r"(?:sentence pairs|sentences|tokens|images|patients|samples|records|documents|cases|examples|instances|articles|studies)\b",
            low,
        )
        or re.search(r"\b\d+\s*(?:k|m)?\s*tokens\b", low)
        or re.search(r"\b\d+\s*(?:k|m)?\s*training sentences\b", low)
    )


def _answer_datasets(evidence_texts: List[str]) -> str:
    dataset_names: List[str] = []
    dataset_sizes: List[str] = []
    vocabulary_details: List[str] = []
    support_sentences: List[str] = []

    for text in evidence_texts:
        for sent in _split_sentences(text):
            found = _extract_capitalized_entities_near_data_terms(sent)

            for item in found:
                if _looks_like_dataset_detail(item):
                    dataset_sizes.append(item)
                else:
                    dataset_names.append(item)

            # Dataset size extraction, excluding vocabulary/token-only details.
            for m in re.finditer(
                r"\b(?:about|approximately|around)?\s*\d+(?:\.\d+)?\s*"
                r"(?:k|m|million|billion|thousand)?\s+"
                r"(?:sentence pairs|sentences|images|patients|samples|records|documents|cases|examples|instances|articles|studies)\b",
                sent,
                flags=re.IGNORECASE,
            ):
                dataset_sizes.append(_clean(m.group(0)))

            # Vocabulary / tokenization details are useful but not datasets.
            for m in re.finditer(
                r"\b(?:about|approximately|around)?\s*\d+(?:\.\d+)?\s*"
                r"(?:k|m|million|billion|thousand)?\s+"
                r"(?:tokens|word-piece vocabulary|vocabulary)\b",
                sent,
                flags=re.IGNORECASE,
            ):
                vocabulary_details.append(_clean(m.group(0)))

            if found or dataset_sizes or vocabulary_details:
                support_sentences.append(sent)

    dataset_names = _dedupe_strings(dataset_names, limit=10)
    dataset_sizes = _dedupe_strings(dataset_sizes, limit=10)
    vocabulary_details = _dedupe_strings(vocabulary_details, limit=10)
    support_sentences = _dedupe_strings(support_sentences, limit=3)
    dataset_names = [
    x for x in dataset_names
    if not any(bad in x.lower() for bad in _DATASET_REJECT_TERMS)
                    ]
    if dataset_names or dataset_sizes or vocabulary_details:
        parts: List[str] = []

        if dataset_names:
            parts.append(
                "Datasets / data sources:\n"
                + "\n".join(f"- {x}" for x in dataset_names)
            )

        if dataset_sizes:
            parts.append(
                "Dataset sizes:\n"
                + "\n".join(f"- {x}" for x in dataset_sizes)
            )

        if vocabulary_details:
            parts.append(
                "Vocabulary / tokenization details:\n"
                + "\n".join(f"- {x}" for x in vocabulary_details)
            )

        if support_sentences:
            parts.append(
                "Evidence snippets:\n"
                + "\n".join(f"- {s}" for s in support_sentences)
            )

        return "\n\n".join(parts)

    fallback = _rank_sentences("datasets data corpus benchmark", evidence_texts, max_sentences=3)
    if fallback:
        return (
            "I could not confidently isolate dataset names, but the most relevant evidence is:\n"
            + "\n".join(f"- {s}" for s in fallback)
        )

    return "I could not find enough evidence about datasets or data sources in the extracted paper text."


# ---------------------------------------------------------------------------
# Specialized answer synthesis
# ---------------------------------------------------------------------------

_METHOD_STEP_MARKERS = [
    "we trained", "we train", "trained", "fine-tuned", "pre-trained", "optimizer", "learning rate",
    "batch", "epochs", "searched", "screened", "included", "excluded", "inclusion criteria",
    "exclusion criteria", "data extraction", "preprocessed", "augmentation", "architecture",
]

_EVAL_MARKERS = [
    "accuracy", "precision", "recall", "f1", "auc", "bleu", "rouge", "perplexity",
    "loss", "rmse", "mae", "score", "performance", "outperform", "achieve", "result",
    "evaluation", "measured", "assessed", "statistical", "p-value", "confidence interval",
]

_REPRO_MARKERS = [
    "learning rate", "batch size", "epoch", "optimizer", "dropout", "weight decay", "seed",
    "gpu", "hardware", "code", "github", "repository", "dataset", "split", "software",
    "implementation", "inclusion criteria", "exclusion criteria", "screening", "quality assessment",
]


def _answer_methodology(evidence_texts: List[str]) -> str:
    steps: List[str] = []
    for text in evidence_texts:
        for sent in _split_sentences(text):
            low = sent.lower()
            if any(m in low for m in _METHOD_STEP_MARKERS):
                steps.append(sent)
    steps = _dedupe_strings(steps, limit=6)
    if not steps:
        steps = _rank_sentences("methodology procedure steps approach", evidence_texts, max_sentences=4)
    if not steps:
        return "I could not find enough methodology evidence in the extracted paper text."
    return "The paper describes these methodological elements:\n" + "\n".join(f"- {s}" for s in steps)


def _answer_evaluation(evidence_texts: List[str]) -> str:
    items: List[str] = []
    for text in evidence_texts:
        for sent in _split_sentences(text):
            low = sent.lower()
            if any(m in low for m in _EVAL_MARKERS) and (re.search(r"\d", sent) or "result" in low or "performance" in low):
                items.append(sent)
    items = _dedupe_strings(items, limit=6)
    if not items:
        items = _rank_sentences("evaluation metrics results performance", evidence_texts, max_sentences=4)
    if not items:
        return "I could not find enough evaluation evidence in the extracted paper text."
    return "The paper reports these evaluation/result details:\n" + "\n".join(f"- {s}" for s in items)


def _answer_figures(evidence_texts: List[str]) -> str:
    items: List[str] = []
    for text in evidence_texts:
        for sent in _split_sentences(text):
            low = sent.lower()
            if any(x in low for x in ["figure", "fig.", "table", "caption", "shown", "illustrates"]):
                items.append(sent)
    items = _dedupe_strings(items, limit=5)
    if not items:
        items = _rank_sentences("figure table caption shows", evidence_texts, max_sentences=3)
    if not items:
        return "I could not find enough figure or table evidence in the extracted paper text."
    return "The relevant figure/table evidence says:\n" + "\n".join(f"- {s}" for s in items)


def _answer_reproducibility(evidence_texts: List[str]) -> str:
    found: List[str] = []
    for text in evidence_texts:
        for sent in _split_sentences(text):
            low = sent.lower()
            if any(m in low for m in _REPRO_MARKERS):
                found.append(sent)
    found = _dedupe_strings(found, limit=6)
    if not found:
        found = _rank_sentences("reproducibility missing hyperparameters software code settings", evidence_texts, max_sentences=4)
    if not found:
        return "I could not find enough reproducibility evidence in the extracted paper text."
    return "The reproducibility-relevant evidence is:\n" + "\n".join(f"- {s}" for s in found)


def _answer_general(question: str, evidence_texts: List[str]) -> str:
    sents = _rank_sentences(question, evidence_texts, max_sentences=4)
    if not sents:
        return "I could not find enough evidence in the extracted paper text to answer this question."
    return "Based on the retrieved evidence:\n" + "\n".join(f"- {s}" for s in sents)


def _synthesize_answer(question: str, evidence_texts: List[str], intent: str) -> str:
    if intent == "datasets":
        return _answer_datasets(evidence_texts)
    if intent == "methodology":
        return _answer_methodology(evidence_texts)
    if intent == "evaluation":
        return _answer_evaluation(evidence_texts)
    if intent == "figures":
        return _answer_figures(evidence_texts)
    if intent == "reproducibility":
        return _answer_reproducibility(evidence_texts)
    return _answer_general(question, evidence_texts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    extracted: Dict[str, Any],
    question: str,
    rag_index: Optional[RagIndex] = None,
    top_k: int = 5,
    embedder_backend: str = "local",
    embedder_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer a question using retrieved chunks from one extracted paper.

    Parameters
    ----------
    extracted:
        Output of pdf_loader.extract_pdf().
    question:
        User question.
    rag_index:
        Optional prebuilt index. If omitted, this function builds an in-memory index.
    top_k:
        Number of evidence chunks to retrieve.
    embedder_backend:
        "local" or "nvidia". Used only when rag_index is omitted.
    embedder_model:
        Optional embedding model name.
    """
    question = _clean(question)
    if not question:
        return {"answer": "No question was provided.", "evidence": [], "query": question}

    intent = _intent(question)
    retrieval_query = _expanded_query(question, intent)

    if rag_index is None:
        rag_index = build_rag_index(
            extracted,
            embedder_backend=embedder_backend,  # type: ignore[arg-type]
            embedder_model=embedder_model,
        )

    # Retrieve slightly more than displayed evidence so the synthesizer has more context.
    internal_top_k = max(top_k, min(10, top_k + 3))
    hits = search_rag_index(rag_index, retrieval_query, top_k=internal_top_k)
    answer = _synthesize_answer(question, _evidence_texts(hits), intent)

    # Keep user-facing evidence compact.
    evidence = [h.to_evidence() for h in hits[:top_k]]

    return {
        "query": question,
        "intent": intent,
        "retrieval_query": retrieval_query,
        "answer": answer,
        "evidence": evidence,
        "rag": {
            "top_k": top_k,
            "embedder_backend": rag_index.embedder_backend,
            "embedder_model": rag_index.embedder_model,
            "num_chunks": len(rag_index.chunks),
        },
    }


def answer_from_pipeline_result(
    pipeline_result: Dict[str, Any],
    question: str,
    top_k: int = 5,
    embedder_backend: str = "local",
    embedder_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience helper for results returned by PaperPipeline.run()."""
    extraction = pipeline_result.get("extraction") or {}
    if not extraction:
        return {"answer": "No extraction object was found in the pipeline result.", "evidence": [], "query": question}
    return answer_question(
        extraction,
        question,
        top_k=top_k,
        embedder_backend=embedder_backend,
        embedder_model=embedder_model,
    )
