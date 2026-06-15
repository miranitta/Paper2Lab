from __future__ import annotations

from typing import Any, Dict, Literal

from paper2lab.inference.nemotron_refiner import refine_with_nemotron


RefinementMode = Literal["none", "nemotron"]


def refine_optional(
    paper_card: Dict[str, Any],
    mode: RefinementMode = "none",
    return_comparison: bool = True,
) -> Dict[str, Any]:
    """
    Default: keep local hardcoded/rule-based extraction.
    Optional: refine the candidate with Nemotron.
    """

    if mode == "none":
        return {
            "status": "skipped",
            "mode": "none",
            "before_refinement": paper_card,
            "after_refinement": paper_card,
            "diff_summary": {
                "changed_fields": [],
                "added_fields": [],
                "removed_fields": [],
            },
        }

    if mode == "nemotron":
        pack = paper_card.get("llm_evidence_pack")

        if not pack:
            return {
                "status": "error",
                "mode": "nemotron",
                "error": "Missing llm_evidence_pack in paper_card.",
                "before_refinement": paper_card,
                "after_refinement": paper_card,
                "diff_summary": {
                    "changed_fields": [],
                    "added_fields": [],
                    "removed_fields": [],
                },
            }

        try:
            return refine_with_nemotron(
                llm_evidence_pack=pack,
                return_comparison=return_comparison,
            )
        except Exception as exc:
            return {
                "status": "error",
                "mode": "nemotron",
                "error": str(exc),
                "before_refinement": paper_card,
                "after_refinement": paper_card,
                "diff_summary": {
                    "changed_fields": [],
                    "added_fields": [],
                    "removed_fields": [],
                },
            }

    raise ValueError(f"Unsupported refinement mode: {mode}")