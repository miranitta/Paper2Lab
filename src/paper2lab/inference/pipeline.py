"""
pipeline.py — Paper2Lab extraction + optional refinement pipeline.

PDF
→ section-aware pdf_loader.extract_pdf()
→ rule-based paper_card
→ local modules
→ optional Nemotron refinement

Default behavior is local-only. Nemotron is optional and safe:
if refinement fails, the local result is still returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from paper2lab.data.pdf_loader import extract_pdf
from paper2lab.evaluation.reproducibility import reproducibility_report
from paper2lab.inference.lab_starter_kit import build_lab_starter_kit
from paper2lab.inference.paper_card import build_paper_card
from paper2lab.inference.refinement import refine_optional
from paper2lab.inference.roadmap import build_reproduction_roadmap
from paper2lab.inference.visual_explainer import explain_figures_and_tables
from paper2lab.inference.auto_select import build_auto_best_card


RefinementMode = Literal["none", "nemotron"]


class PaperPipeline:
    def __init__(
        self,
        pdf_engine: str = "pymupdf",
        include_extraction: bool = True,
        include_llm_pack: bool = True,
        include_local_modules: bool = True,
        refinement_mode: RefinementMode = "none",
    ) -> None:
        self.pdf_engine = pdf_engine
        self.include_extraction = include_extraction
        self.include_llm_pack = include_llm_pack
        self.include_local_modules = include_local_modules
        self.refinement_mode = refinement_mode

    def run(
        self,
        pdf_path: str | Path,
        refinement_mode: RefinementMode | None = None,
    ) -> Dict[str, Any]:
        active_refinement_mode = refinement_mode or self.refinement_mode

        extracted = extract_pdf(pdf_path, engine=self.pdf_engine)
        paper_card = build_paper_card(extracted)

        if self.include_local_modules:
            reproduction_roadmap = build_reproduction_roadmap(extracted, paper_card)
            figures_and_tables = explain_figures_and_tables(extracted)

            paper_card["methodology_steps"] = reproduction_roadmap.get("experimental_steps", [])
            paper_card["reproduction_roadmap"] = reproduction_roadmap
            paper_card["figures_and_tables"] = figures_and_tables
            paper_card["reproducibility_score"] = reproducibility_report(extracted, paper_card)
            paper_card["lab_starter_kit"] = build_lab_starter_kit(paper_card)

            # Keep the LLM evidence pack aligned with the final local candidate.
            if "llm_evidence_pack" in paper_card:
                paper_card["llm_evidence_pack"]["candidate_paper_card"] = {
                    k: v for k, v in paper_card.items()
                    if k != "llm_evidence_pack"
                }

        refinement = refine_optional(
            paper_card=paper_card,
            mode=active_refinement_mode,
            return_comparison=True,
        )
        auto_selection = build_auto_best_card(
        local_card=paper_card,
        refinement=refinement,
            )

        final_paper_card = auto_selection["final_paper_card"]

        refined_card = refinement.get("after_refinement", paper_card)
        if not isinstance(refined_card, dict):
            refined_card = paper_card

        if not self.include_llm_pack:
            paper_card = {
                k: v for k, v in paper_card.items()
                if k != "llm_evidence_pack"
            }
            refined_card = {
                k: v for k, v in refined_card.items()
                if k != "llm_evidence_pack"
            }

            if isinstance(refinement.get("before_refinement"), dict):
                refinement["before_refinement"] = {
                    k: v for k, v in refinement["before_refinement"].items()
                    if k != "llm_evidence_pack"
                }

            if isinstance(refinement.get("after_refinement"), dict):
                refinement["after_refinement"] = {
                    k: v for k, v in refinement["after_refinement"].items()
                    if k != "llm_evidence_pack"
                }

        result: Dict[str, Any] = {
            "status": "ok",
            "refinement_mode": active_refinement_mode,
            "paper_card": paper_card,
            "paper_card_refined": refinement.get("after_refinement", paper_card),
            "paper_card_final": final_paper_card,
            "refinement": refinement,
            "auto_selection": auto_selection,
}

        if self.include_extraction:
            result["extraction"] = {
                "source_pdf": extracted.get("source_pdf"),
                "num_pages": extracted.get("num_pages"),
                "title": extracted.get("title"),
                "abstract": extracted.get("abstract"),
                "extraction_engine": extracted.get("extraction_engine"),
                "quality": extracted.get("quality", {}),
                "metadata": extracted.get("metadata", {}),
                "sections": extracted.get("sections", []),
                "all_sections": extracted.get("all_sections", []),
                "references": extracted.get("references", []),
                "references_text_preview": (extracted.get("references_text") or "")[:2000],
                "appendix_text_preview": (extracted.get("appendix_text") or "")[:1500],
                "boilerplate_text_preview": (extracted.get("boilerplate_text") or "")[:1500],
                "captions": extracted.get("captions", []),
                "tables": extracted.get("tables", []),
                "clean_text_preview": (
                    extracted.get("clean_text")
                    or extracted.get("text")
                    or ""
                )[:3000],
                "raw_text_preview": (extracted.get("raw_text") or "")[:3000],
                "text_preview": (
                    extracted.get("clean_text")
                    or extracted.get("text")
                    or ""
                )[:3000],
            }

        return result

    def run_batch(
        self,
        pdf_paths: List[str | Path],
        refinement_mode: RefinementMode | None = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for path in pdf_paths:
            try:
                results.append(
                    self.run(
                        path,
                        refinement_mode=refinement_mode,
                    )
                )
            except Exception as exc:
                results.append({
                    "status": "error",
                    "source_pdf": str(path),
                    "error": str(exc),
                    "paper_card": None,
                    "paper_card_refined": None,
                    "refinement": {
                        "status": "error",
                        "mode": refinement_mode or self.refinement_mode,
                        "error": str(exc),
                    },
                    "extraction": None,
                })

        return results

    def save_json(
        self,
        pdf_path: str | Path,
        output_path: str | Path,
        refinement_mode: RefinementMode | None = None,
    ) -> None:
        result = self.run(
            pdf_path,
            refinement_mode=refinement_mode,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)