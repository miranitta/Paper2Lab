"""
gradio_pipeline.py — Gradio UI for Paper2Lab section-aware extraction.

This UI matches the current pre-LLM pipeline:
- No Anthropic parameters.
- Shows section roles and whether references/appendix were removed from clean text.
- Downloads the paper_card JSON.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, Tuple

import gradio as gr

from paper2lab.inference.pipeline import PaperPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_list(items: list[str] | None, max_items: int = 8) -> str:
    if not items:
        return "_None detected_"
    shown = items[:max_items]
    suffix = f"\n… +{len(items) - max_items} more" if len(items) > max_items else ""
    return "\n".join(f"- {s}" for s in shown) + suffix


def _quality_badge(score: float) -> str:
    if score >= 0.75:
        return f"🟢 Quality score: {score:.2f}"
    if score >= 0.45:
        return f"🟡 Quality score: {score:.2f}"
    return f"🔴 Quality score: {score:.2f} — extraction may be incomplete"


def _build_overview(result: Dict[str, Any]) -> str:
    card = result["paper_card"]
    ext = result["extraction"]
    quality = ext.get("quality", {})
    metadata = card.get("metadata", {})

    lines = [
        f"## {card.get('title') or '_(title not found)_'}",
        f"**Field:** {card.get('field', '—')}  |  "
        f"**Pages:** {ext.get('num_pages', '?')}  |  "
        f"**Engine:** {ext.get('extraction_engine', '?')}  |  "
        + _quality_badge(float(quality.get("quality_score", 0.0))),
        "",
        "### Extraction Safety",
        f"- References removed from body text: {'✅' if metadata.get('references_removed_from_body') else '⚠️ not detected'}",
        f"- Appendix removed from body text: {'✅' if metadata.get('appendix_removed_from_body') else '—'}",
        f"- Methodology section found: {'✅' if quality.get('methodology_section_found') else '⚠️ fallback may be used'}",
        "",
        "### Research Question",
        card.get("research_question") or "_Not detected_",
        "",
        "### Abstract",
        ext.get("abstract") or "_Not extracted_",
        "",
        "### Contributions",
        _fmt_list(card.get("contributions")),
        "",
        "### Methodology",
        _fmt_list(card.get("methodology")),
        "",
        "### Datasets / Data Sources",
        _fmt_list(card.get("datasets_or_data_sources")),
        "",
        "### Models / Methods",
        _fmt_list(card.get("models_or_methods")),
        "",
        "### Metrics & Measurements",
        _fmt_list(card.get("metrics_or_measurements")),
        "",
        "### Key Findings",
        _fmt_list(card.get("key_findings")),
        "",
        "### Limitations",
        _fmt_list(card.get("limitations")),
        "",
        "### Missing Reproducibility Info",
        _fmt_list(card.get("missing_reproducibility_info")),
    ]
    return "\n".join(lines)


def _build_extraction_details(result: Dict[str, Any]) -> str:
    ext = result["extraction"]
    sections = ext.get("sections", [])
    refs = ext.get("references", [])
    captions = ext.get("captions", [])
    tables = ext.get("tables", [])
    quality = ext.get("quality", {})

    section_list = "\n".join(
        f"  - **{s.get('title', '?')}** — role `{s.get('role', 'other')}`, "
        f"pages {s.get('page_start', '?')}–{s.get('page_end', '?')}, "
        f"{len((s.get('text') or '').split())} words"
        for s in sections
    )
    ref_sample = "\n".join(f"  {i + 1}. {r[:140]}…" for i, r in enumerate(refs[:5]))
    cap_sample = "\n".join(
        f"  - **{c.get('label')}**: {(c.get('caption') or '')[:120]}…" for c in captions[:5]
    )
    table_info = "\n".join(
        f"  - Page {t.get('page_number', '?')}, {len(t.get('data', []))} rows × "
        f"{len(t.get('data', [[]])[0]) if t.get('data') else '?'} cols"
        for t in tables[:5]
    )

    lines = [
        "## Extraction Details",
        "",
        "### Quality",
        f"- Title found: {'✅' if quality.get('title_found') else '❌'}",
        f"- Abstract found: {'✅' if quality.get('abstract_found') else '❌'}",
        f"- Sections: {quality.get('num_sections', 0)}",
        f"- Section roles: `{', '.join(quality.get('section_roles', []))}`",
        f"- References: {quality.get('num_references', 0)}",
        f"- Captions: {quality.get('num_captions', 0)}",
        f"- Tables: {quality.get('num_tables', 0)}",
        "",
        "### Sections Detected",
        section_list or "_None_",
        "",
        "### References moved to metadata/body-excluded area — first 5",
        ref_sample or "_None_",
        "",
        "### Captions — first 5",
        cap_sample or "_None_",
        "",
        "### Tables — first 5",
        table_info or "_None_",
        "",
        "### Clean Text Preview — references excluded",
        "```",
        ext.get("text_preview", "")[:1800],
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_pdf(pdf_file: Any, engine: str, include_llm_pack: bool) -> Tuple[str, str, str, str]:
    if pdf_file is None:
        return "", "", "", "⚠️ Please upload a PDF first."

    pipeline = PaperPipeline(
        pdf_engine=engine,
        include_extraction=True,
        include_llm_pack=include_llm_pack,
    )

    try:
        result = pipeline.run(pdf_file.name if hasattr(pdf_file, "name") else pdf_file)
    except Exception as exc:
        return "", "", "", f"❌ Error: {exc}"

    overview = _build_overview(result)
    details = _build_extraction_details(result)
    card_preview = {
        k: v for k, v in result["paper_card"].items()
        if k != "llm_evidence_pack"
    }
    json_preview = json.dumps(card_preview, indent=2, ensure_ascii=False)
    score = result["extraction"].get("quality", {}).get("quality_score", 0.0)
    status = f"✅ Done — section-aware extraction quality: {score:.2f}"
    return overview, details, json_preview, status


def download_json(pdf_file: Any, engine: str, include_llm_pack: bool) -> str | None:
    if pdf_file is None:
        return None
    pipeline = PaperPipeline(
        pdf_engine=engine,
        include_extraction=True,
        include_llm_pack=include_llm_pack,
    )
    try:
        result = pipeline.run(pdf_file.name if hasattr(pdf_file, "name") else pdf_file)
    except Exception:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8")
    json.dump(result["paper_card"], tmp, indent=2, ensure_ascii=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Paper2Lab", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📄 Paper2Lab — Section-Aware Academic Paper Extractor")
        gr.Markdown(
            "Upload a research paper PDF. The pipeline detects section headers, removes references from body text, "
            "and builds a structured paper card ready for later Nemotron refinement."
        )

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                engine = gr.Radio(
                    choices=["pymupdf", "docling", "auto"],
                    value="pymupdf",
                    label="Extraction engine",
                    info="pymupdf = fast; docling = optional complex-layout engine; auto = compare quality",
                )
                include_llm_pack = gr.Checkbox(
                    label="Include llm_evidence_pack",
                    value=True,
                    info="Useful for later Nemotron/LLM refinement; turn off for simpler JSON.",
                )
                run_btn = gr.Button("▶ Extract", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False)
                download_btn = gr.Button("⬇ Download Paper Card JSON")
                download_file = gr.File(label="JSON download", interactive=False)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📋 Paper Card"):
                        overview_md = gr.Markdown()
                    with gr.Tab("🔬 Extraction Details"):
                        details_md = gr.Markdown()
                    with gr.Tab("{ } JSON Preview"):
                        json_box = gr.Code(language="json", interactive=False)

        run_btn.click(
            fn=process_pdf,
            inputs=[pdf_input, engine, include_llm_pack],
            outputs=[overview_md, details_md, json_box, status_box],
        )
        download_btn.click(
            fn=download_json,
            inputs=[pdf_input, engine, include_llm_pack],
            outputs=download_file,
        )
    return demo


pipeline = PaperPipeline(pdf_engine="pymupdf")


def process_pdf_simple(pdf_file: Any) -> Dict[str, Any]:
    if pdf_file is None:
        return {"error": "No PDF uploaded"}
    return pipeline.run(pdf_file.name if hasattr(pdf_file, "name") else pdf_file)


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False)
