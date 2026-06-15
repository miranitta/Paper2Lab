import json
import tempfile
import html
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

print("DEBUG ROOT_DIR:", ROOT_DIR)
print("DEBUG SRC_DIR:", SRC_DIR)
print("DEBUG SRC_EXISTS:", SRC_DIR.exists())
print("DEBUG PIPELINE_EXISTS:", (SRC_DIR / "paper2lab" / "inference" / "pipeline.py").exists())
print("DEBUG sys.path[:5]:", sys.path[:5])

import gradio as gr

try:
    from paper2lab.inference.pipeline import PaperPipeline
    print("DEBUG PaperPipeline import: OK")
except Exception as e:
    print("DEBUG PaperPipeline import failed:", repr(e))
    PaperPipeline = None

try:
    from paper2lab.rag.qa import answer_from_pipeline_result
    print("DEBUG RAG import: OK")
except Exception as e:
    print("DEBUG RAG import failed:", repr(e))
    answer_from_pipeline_result = None


APP_NAME = "Paper2Lab"


def pretty_json(data):
    return json.dumps(data or {}, indent=2, ensure_ascii=False)


def as_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    return [value]


def bullet_list(items, empty="_Not found._"):
    items = as_list(items)
    if not items:
        return empty
    return "\n".join(f"- {x}" for x in items)


def clean_display_text(text, max_len=95):
    text = str(text or "").replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text


def chip_list(items, empty="Not found"):
    items = as_list(items)
    if not items:
        return f"<span class='muted'>{empty}</span>"

    html = ""
    for x in items[:6]:
        clean = clean_display_text(x, max_len=70)
        html += f"<span class='chip' title='{str(x)}'>{clean}</span>"
    return html


def get_card(result):
    return result.get("paper_card_final") or result.get("paper_card") or result


def quality_warnings(card):
    warnings = []

    rq = card.get("research_question", "")
    if rq and ("BERTis" in rq or len(rq.split()) < 5):
        warnings.append("Research question may contain PDF spacing noise.")

    datasets = as_list(card.get("datasets_or_data_sources"))
    if any(len(str(x)) > 220 for x in datasets):
        warnings.append("Dataset extraction contains long noisy candidates.")

    for field in ["contributions", "key_findings", "limitations"]:
        if not card.get(field):
            warnings.append(f"{field} is empty.")

    return warnings


def build_quick_summary_html(result):
    card = get_card(result)
    warnings = quality_warnings(card)

    status = "Ready"
    if result:
        status = f"{max(0, 6 - len(warnings))} / 6 fields complete"

    return f"""
    <div class="summary-card">
      <div class="card-head">
        <div><span class="icon">📄</span><b>Quick summary</b></div>
        <span class="status-pill">{status}</span>
      </div>

      <div class="summary-row">
        <div class="label">Title</div>
        <div class="value">{card.get("title", "Untitled paper")}</div>
      </div>

      <div class="summary-row">
        <div class="label">Research question</div>
        <div class="value">{card.get("research_question", "Not found")}</div>
      </div>

      <div class="summary-row">
        <div class="label">Datasets</div>
        <div class="value">{chip_list(card.get("datasets_or_data_sources"))}</div>
      </div>

      <div class="summary-row">
        <div class="label">Models</div>
        <div class="value">{chip_list(card.get("models_or_methods"))}</div>
      </div>

      <div class="summary-row">
        <div class="label">Key findings</div>
        <div class="value highlight">{bullet_list(card.get("key_findings")).replace("- ", "• ")}</div>
      </div>
    </div>
    """


def build_paper_summary_md(result):
    card = get_card(result)
    warnings = quality_warnings(card)

    warning_md = ""
    if warnings:
        warning_md = "## ⚠️ Extraction Quality Warnings\n" + "\n".join(f"- {w}" for w in warnings)

    return f"""
# Structured Paper Summary

**Title:** {card.get("title", "Untitled paper")}  
**Field:** {card.get("field", "Unknown")}

{warning_md}

## Research Question
{card.get("research_question", "_Not found._")}

## Contributions
{bullet_list(card.get("contributions"))}

## Methodology
{bullet_list(card.get("methodology"))}

## Datasets / Data Sources
{bullet_list(card.get("datasets_or_data_sources"))}

## Models / Methods
{bullet_list(card.get("models_or_methods"))}

## Metrics / Measurements
{bullet_list(card.get("metrics_or_measurements"))}

## Key Findings
{bullet_list(card.get("key_findings"))}

## Limitations
{bullet_list(card.get("limitations"))}
"""


def build_lab_md(result):
    card = get_card(result)
    kit = card.get("lab_starter_kit") or result.get("lab_starter_kit") or {}

    return f"""
# Lab Starter Kit

**Starter type:** `{kit.get("starter_type", card.get("paper_type", "unknown"))}`

## Project Structure
{bullet_list(kit.get("project_structure"))}

## Requirements
{bullet_list(kit.get("requirements_txt"))}

## Dataset Plan
{bullet_list(kit.get("dataset_plan") or kit.get("required_data"))}

## Suggested Experiments
{bullet_list(kit.get("suggested_experiments") or kit.get("experiment_checklist"))}

## Evaluation Plan
{bullet_list(kit.get("evaluation_plan"))}

## Reproducibility Risks
{bullet_list(kit.get("reproducibility_risks") or card.get("missing_reproducibility_info"))}
"""


def build_evidence_md(result):
    figs = result.get("figures_and_tables", [])
    card = get_card(result)

    lines = ["# Evidence Viewer"]

    if figs:
        lines.append("## Figures and Tables")
        for item in figs[:12]:
            lines.append(
                f"- **{item.get('label', 'Item')}** — page {item.get('page_number', '?')}: "
                f"{item.get('summary', 'No summary')}"
            )

    lines.append("\n## Extracted Evidence Fields")
    for field in [
        "methodology",
        "datasets_or_data_sources",
        "models_or_methods",
        "metrics_or_measurements",
        "key_findings",
    ]:
        lines.append(f"\n### {field}")
        lines.append(bullet_list(card.get(field)))

    return "\n".join(lines)


def build_advanced_md(result):
    card = get_card(result)

    metadata = card.get("metadata", {}) if isinstance(card, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    quality = metadata.get("quality", {})
    if not isinstance(quality, dict):
        quality = {}

    repro = card.get("reproducibility_score", {}) if isinstance(card, dict) else {}
    if not isinstance(repro, dict):
        repro = {}

    selection = result.get("auto_selection", {}).get("selection_report", {})
    if not isinstance(selection, dict):
        selection = {}

    md = []
    md.append("# ⚙️ Advanced Analysis")
    md.append("")
    md.append("## Extraction Quality")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Quality Score | {quality.get('quality_score', 'N/A')} |")
    md.append(f"| Sections Found | {quality.get('num_sections', 'N/A')} |")
    md.append(f"| References Found | {quality.get('num_references', metadata.get('references_count', 'N/A'))} |")
    md.append(f"| Tables Found | {quality.get('num_tables', 'N/A')} |")
    md.append(f"| Captions Found | {quality.get('num_captions', 'N/A')} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"**Reproducibility Level:** {repro.get('level', 'unknown')}")
    md.append("")
    md.append(f"**Score:** {repro.get('score', 'N/A')}")
    md.append("")
    md.append("### Detected Items")
    md.append(bullet_list(repro.get("detected_items")))
    md.append("")
    md.append("### Missing Items")
    md.append(bullet_list(repro.get("missing_items")))
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Selection Report")

    if selection.get("fields"):
      md.append("Field-level local vs Nemotron selection was completed.")
      md.append("")
      md.append("```json")
      md.append(pretty_json(selection))
      md.append("```")
    else:
      md.append(
        "No field-level comparison available for this paper. "
        "The final card was generated from the best available extraction pipeline."
      )

    return "\n".join(md)


def safe_html(value):
    return html.escape(str(value or ""), quote=True)


def short_card_text(value, max_len=48):
    text = clean_display_text(value, max_len=max_len)
    return safe_html(text)


def build_lab_cards_html(result):
    card = get_card(result)
    kit = card.get("lab_starter_kit") or result.get("lab_starter_kit") or {}

    reqs = as_list(kit.get("requirements_txt"))
    risks = as_list(
        kit.get("reproducibility_risks")
        or card.get("missing_reproducibility_info")
    )
    experiments = as_list(
        kit.get("suggested_experiments")
        or kit.get("experiment_checklist")
    )

    starter_type = kit.get("starter_type", card.get("paper_type", "unknown"))
    tools = ", ".join(clean_display_text(x, max_len=28) for x in reqs[:3]) if reqs else "Not found"

    return f"""
    <div class="lab-board">
      <div class="lab-board-header">
        <span class="lab-emoji">🧪</span>
        <span>Lab readiness</span>
      </div>

      <div class="lab-grid-2">
        <div class="lab-tile lab-green">
          <span>Starter type</span>
          <b>{short_card_text(starter_type)}</b>
        </div>

        <div class="lab-tile lab-dark">
          <span>Tools</span>
          <b>{short_card_text(tools, max_len=58)}</b>
        </div>

        <div class="lab-tile lab-purple">
          <span>Experiments</span>
          <b>{len(experiments)} detected</b>
        </div>

        <div class="lab-tile lab-red">
          <span>Risks</span>
          <b>{len(risks)} issues</b>
        </div>
      </div>
    </div>
    """

def analyze_paper(pdf_file, refinement_mode):
    if pdf_file is None:
        raise gr.Error("Please upload a PDF first.")

    if PaperPipeline is None:
        raise gr.Error("PaperPipeline import failed. Run from the project root or install your package.")

    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    pipeline = PaperPipeline(refinement_mode=refinement_mode)
    result = pipeline.run(pdf_path)
    print("PDF PATH:", pdf_path)
    print("MODE:", refinement_mode)
    print("FINAL DATASETS:", result.get("paper_card_final", {}).get("datasets_or_data_sources"))
    print("GET_CARD DATASETS:", get_card(result).get("datasets_or_data_sources"))

    card = get_card(result)

    if not card.get("datasets_or_data_sources"):
      roadmap = card.get("reproduction_roadmap") or {}
      kit = card.get("lab_starter_kit") or {}

      fallback_datasets = []

      if isinstance(roadmap, dict):
          fallback_datasets += roadmap.get("datasets") or []

      if isinstance(kit, dict):
          fallback_datasets += kit.get("dataset_plan") or []

      if fallback_datasets:
          card["datasets_or_data_sources"] = list(dict.fromkeys(fallback_datasets))
          result["paper_card_final"] = card

    paper_md = build_paper_summary_md(result)
    lab_md = build_lab_md(result)
    evidence_md = build_evidence_md(result)
    advanced_md = build_advanced_md(result)
    quick_html = build_quick_summary_html(result)
    lab_cards = build_lab_cards_html(result)

    tmp = Path(tempfile.mkdtemp())
    json_path = tmp / "paper2lab_output.json"
    md_path = tmp / "paper2lab_report.md"

    json_path.write_text(pretty_json(result), encoding="utf-8")
    md_path.write_text(paper_md + "\n\n---\n\n" + lab_md, encoding="utf-8")

    return (
        result,
        quick_html,
        paper_md,
        lab_md,
        evidence_md,
        advanced_md,
        lab_cards,
        card,
        str(json_path),
        str(md_path),
    )

def ask_paper_question(result, question):
    if not result:
        return "⚠️ Analyze a paper first.", {}

    if not question or not str(question).strip():
        return "⚠️ Please enter a question.", {}

    question_text = str(question).strip()
    q = question_text.lower()

    card = get_card(result)

    def md_items(title, items):
        items = as_list(items)
        items = [x for x in items if str(x).strip()]
        if not items:
            return f"**{title}:**\n\n_Not found in the structured paper card._"

        return f"**{title}:**\n\n" + "\n".join(f"- {x}" for x in items[:8])

    # Fast structured answers for demo-critical questions
    if any(k in q for k in ["dataset", "data source", "corpus", "benchmark"]):
        return md_items("Datasets / data sources", card.get("datasets_or_data_sources")), {
            "source": "structured_card",
            "field": "datasets_or_data_sources",
        }

    if any(k in q for k in ["model", "architecture", "proposed"]):

      rq = card.get("research_question", "")

      if rq:
          return (
              f"**Proposed model:**\n\n- {rq}",
            {
                "source": "structured_card",
                "field": "research_question",
            },
        )

    if any(k in q for k in ["finding", "result", "conclusion", "contribution"]):
        return md_items("Key findings", card.get("key_findings") or card.get("contributions")), {
            "source": "structured_card",
            "field": "key_findings",
        }

    if any(k in q for k in ["metric", "score", "performance", "evaluation"]):
        return md_items("Metrics / measurements", card.get("metrics_or_measurements")), {
            "source": "structured_card",
            "field": "metrics_or_measurements",
        }

    if any(k in q for k in ["limitation", "risk", "missing"]):
        return md_items("Limitations / reproducibility risks", card.get("limitations")), {
            "source": "structured_card",
            "field": "limitations",
        }

    if any(k in q for k in ["reproduce", "reproduction", "roadmap", "steps"]):
        roadmap = card.get("reproduction_roadmap") or {}
        steps = []
        if isinstance(roadmap, dict):
            steps = roadmap.get("experimental_steps") or roadmap.get("missing_for_reproduction") or []
        return md_items("Reproduction roadmap", steps), {
            "source": "structured_card",
            "field": "reproduction_roadmap",
        }

    if answer_from_pipeline_result is None:
        return (
            "⚠️ RAG module import failed. Check that `paper2lab.rag.qa` is available "
            "and dependencies are installed.",
            {},
        )

    try:
        qa = answer_from_pipeline_result(
            pipeline_result=result,
            question=question_text,
            top_k=5,
            embedder_backend="local",
        )

        answer = qa.get("answer", "No answer found.")
        evidence = qa.get("evidence", [])

        evidence_md = "\n\n## Evidence\n"
        for i, ev in enumerate(evidence, 1):
            page = ev.get("page_start") or ev.get("page_number") or "?"
            title = ev.get("title", "Evidence")
            text = ev.get("text", "")
            evidence_md += f"\n**{i}. {title} — page {page}**\n\n> {text[:700]}\n"

        return answer + evidence_md, qa

    except Exception as exc:
        return f"❌ RAG error: {exc}", {}


CSS = """
/* ─────────────────────────────────────────────
   Paper2Lab UI — fixed layout
   Fixes:
   - no duplicate upload holder/native file input
   - wider hero with Paper2Lab title
   - clean 2-column app layout
   - lab card placed under Quick Summary
───────────────────────────────────────────── */

body,
.gradio-container {
  background: #eef2f7 !important;
  color: #111827 !important;
}

* {
  box-sizing: border-box;
}

footer {
  display: none !important;
}

.gradio-container {
  max-width: none !important;
  padding-top: 28px !important;
}

/* Main page width */
.app-shell {
  width: min(1220px, 94vw);
  margin: 0 auto;
}

/* Remove unwanted Gradio block styling only inside our custom layout */
.top-layout .block,
.top-layout .gr-group,
.top-layout .gr-box,
.top-layout .gr-panel,
.result-stack .block,
.result-stack .gr-group,
.result-stack .gr-box,
.result-stack .gr-panel {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: visible !important;
}

/* ───────────────── HERO ───────────────── */

.hero {
  position: relative;
  overflow: hidden;
  min-height: 245px;
  border-radius: 24px;
  padding: 34px 48px;
  margin: 0 auto 28px;
  background: linear-gradient(135deg, #0f172a 0%, #18253f 48%, #252659 100%);
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 22px 60px rgba(15,23,42,.18);
}

.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 15% 25%, rgba(34,211,238,.14), transparent 30%),
    radial-gradient(circle at 78% 26%, rgba(124,58,237,.22), transparent 35%);
  pointer-events: none;
}

.hero-content {
  position: relative;
  z-index: 2;
  max-width: 920px;
}

.logo-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}

.logo-mark {
  width: 32px;
  height: 32px;
  display: inline-grid;
  place-items: center;
  border-radius: 10px;
  background: rgba(103,232,249,.14);
  border: 1px solid rgba(103,232,249,.22);
  font-size: 18px;
}

.logo-text {
  color: #ffffff !important;
  font-size: 18px;
  font-weight: 950;
  letter-spacing: -0.035em;
}

.kicker {
  color: #67e8f9 !important;
  text-transform: uppercase;
  letter-spacing: .16em;
  font-size: 11px;
  font-weight: 900;
  margin-bottom: 8px;
}

.hero h1 {
  max-width: 960px;
  color: #ffffff !important;
  font-size: 38px;
  line-height: 1.05;
  letter-spacing: -0.052em;
  margin: 0 0 14px;
}

.hero h1 span {
  color: #a78bfa !important;
}

.hero p {
  max-width: 760px;
  color: rgba(255,255,255,.86) !important;
  font-size: 15px;
  line-height: 1.62;
  margin: 0 0 22px;
}

.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.hero-badge {
  padding: 8px 13px;
  border-radius: 10px;
  background: rgba(255,255,255,.12);
  color: #ffffff !important;
  font-weight: 850;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,.18);
}

/* ───────────────── TWO COLUMN LAYOUT ───────────────── */

.top-layout {
  display: grid !important;
  grid-template-columns: 340px minmax(0, 1fr) !important;
  gap: 28px !important;
  align-items: start !important;
}

.control-stack {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 24px !important;
  padding: 24px !important;
  box-shadow: 0 14px 38px rgba(15,23,42,.06) !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 18px !important;
}

.result-stack {
  display: flex !important;
  flex-direction: column !important;
  gap: 26px !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

/* ───────────────── LEFT PANEL ───────────────── */

.panel-title {
  font-size: 16px;
  font-weight: 900;
  color: #111827 !important;
  margin: 0;
  padding-bottom: 10px;
  border-bottom: 1px solid #f1f5f9;
}

/* Button */
.gradio-container button.primary,
.gradio-container button[class*="primary"] {
  width: 100% !important;
  min-height: 50px !important;
  border: none !important;
  border-radius: 15px !important;
  background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
  color: #ffffff !important;
  font-weight: 950 !important;
  font-size: 15px !important;
  box-shadow: 0 12px 26px rgba(124,58,237,.28) !important;
}

.gradio-container button.primary *,
.gradio-container button[class*="primary"] * {
  color: #ffffff !important;
}

.tip-box {
  padding: 12px 14px;
  border-radius: 14px;
  background: #fafbff;
  border: 1px solid #e5e7eb;
  color: #4b5563 !important;
  font-size: 12px;
  font-weight: 750;
}

/* ───────────────── QUICK SUMMARY ───────────────── */

.summary-card {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 22px !important;
  overflow: hidden !important;
  box-shadow: 0 14px 38px rgba(15,23,42,.06) !important;
}

.card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 17px 22px;
  border-bottom: 1px solid #f1f5f9;
}

.card-head b {
  color: #111827 !important;
  font-size: 16px;
}

.icon {
  display: inline-grid;
  place-items: center;
  width: 28px;
  height: 28px;
  margin-right: 9px;
  border-radius: 9px;
  background: #ede9fe;
  font-size: 15px;
}

.status-pill {
  background: #dcfce7;
  color: #047857 !important;
  padding: 6px 13px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 12px;
}

.summary-row {
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr);
  gap: 16px;
  padding: 15px 22px;
  border-bottom: 1px solid #f8fafc;
  font-size: 14px;
}

.label {
  color: #6b7280 !important;
  font-weight: 850;
}

.value {
  color: #111827 !important;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.highlight {
  background: #f5f3ff;
  border-left: 3px solid #7c3aed;
  border-radius: 0 10px 10px 0;
  padding: 10px 12px;
  color: #312e81 !important;
}

.chip {
  display: inline-block;
  margin: 3px 5px 3px 0;
  padding: 6px 10px;
  border-radius: 999px;
  background: #ede9fe;
  color: #4c1d95 !important;
  border: 1px solid #c4b5fd;
  font-weight: 800;
  font-size: 12px;
}

.muted {
  color: #9ca3af !important;
}

/* ───────────────── LAB READINESS ───────────────── */

.lab-board {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 22px !important;
  padding: 18px !important;
  box-shadow: 0 14px 38px rgba(15,23,42,.06) !important;
}

.lab-board-header {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #f8f9fc;
  border: 1px solid #ebebf0;
  border-radius: 14px;
  padding: 14px 18px;
  margin-bottom: 16px;
  color: #0f172a !important;
  font-size: 18px;
  font-weight: 950;
  letter-spacing: -0.03em;
}

.lab-board-header * {
  color: #0f172a !important;
}

.lab-emoji {
  font-size: 22px;
  line-height: 1;
  flex: 0 0 auto;
}

.lab-grid-2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.lab-tile {
  min-height: 108px;
  padding: 18px 20px;
  border-radius: 16px;
  border: 1px solid transparent;
  overflow: hidden;
}

.lab-tile span {
  display: block;
  margin-bottom: 10px;
  font-size: 12px;
  line-height: 1.2;
  font-weight: 900;
  letter-spacing: .04em;
  text-transform: uppercase;
}

.lab-tile b {
  display: block;
  font-size: 23px;
  line-height: 1.15;
  font-weight: 950;
  letter-spacing: -0.035em;
  overflow-wrap: break-word;
}

.lab-green {
  background: #ecfdf5 !important;
  color: #065f46 !important;
  border-color: #a7f3d0 !important;
}

.lab-dark {
  background:
    radial-gradient(circle at 84% 18%, rgba(124,58,237,.28), transparent 36%),
    #1e1b4b !important;
  color: #ffffff !important;
  border-color: #3730a3 !important;
}

.lab-purple {
  background: #f5f3ff !important;
  color: #4c1d95 !important;
  border-color: #ddd6fe !important;
}

.lab-red {
  background: #fff1f2 !important;
  color: #9f1239 !important;
  border-color: #fecdd3 !important;
}

.lab-green *,
.lab-dark *,
.lab-purple *,
.lab-red * {
  color: inherit !important;
}

/* ───────────────── TABS ───────────────── */

.tabs-shell {
  margin-top: 28px;
}

.gradio-container .tab-nav {
  gap: 6px !important;
  border-bottom: 1px solid #cbd5e1 !important;
}

.gradio-container .tab-nav button {
  font-size: 14px !important;
  font-weight: 750 !important;
  color: #6b7280 !important;
  padding: 11px 18px !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  white-space: nowrap !important;
}

.gradio-container .tab-nav button.selected {
  color: #6d28d9 !important;
  border-bottom-color: #7c3aed !important;
}

.gradio-container .tabs > .block,
.gradio-container .tabitem {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 18px !important;
  box-shadow: 0 8px 24px rgba(15,23,42,.04) !important;
  overflow: hidden !important;
}

/* General text */
.gradio-container .markdown,
.gradio-container p,
.gradio-container li,
.gradio-container td,
.gradio-container th,
.gradio-container label,
.gradio-container textarea,
.gradio-container input {
  color: #111827 !important;
}

/* ───────────────── RESPONSIVE ───────────────── */

@media (max-width: 900px) {
  .app-shell {
    width: min(94vw, 760px);
  }

  .hero {
    min-height: auto;
    padding: 26px 22px;
  }

  .hero h1 {
    font-size: 28px;
  }

  .hero p {
    font-size: 14px;
  }

  .top-layout {
    grid-template-columns: 1fr !important;
  }

  .summary-row {
    grid-template-columns: 1fr;
    gap: 6px;
  }

  .lab-grid-2 {
    grid-template-columns: 1fr;
  }
}


/* "Analyze Paper" text inside Quick Summary */
.summary-card .value,
.summary-card .value *,
.summary-row .value,
.summary-row .value * {
  color: #111827 !important;
  opacity: 1 !important;
  visibility: visible !important;
}

/* Quick summary text */
.summary-card .value,
.summary-card .value *,
.summary-row .value,
.summary-row .value * {
  color: #111827 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #111827 !important;
}

/* Tabs text */
.gradio-container button[role="tab"],
.gradio-container button[role="tab"] *,
.gradio-container .tab-nav button,
.gradio-container .tab-nav button * {
  color: #111827 !important;
  opacity: 1 !important;
  visibility: visible !important;
  -webkit-text-fill-color: #111827 !important;
}

.gradio-container button[role="tab"][aria-selected="true"],
.gradio-container button[role="tab"][aria-selected="true"] *,
.gradio-container .tab-nav button.selected,
.gradio-container .tab-nav button.selected * {
  color: #7c3aed !important;
  -webkit-text-fill-color: #7c3aed !important;
  font-weight: 800 !important;
}

/* ===== FINAL LEFT PANEL UPLOAD FIX ===== */

.upload-caption {
  display: inline-flex;
  width: fit-content;
  align-items: center;
  gap: 8px;
  padding: 11px 20px;
  border-radius: 12px;
  background: linear-gradient(135deg, #7c3aed, #6d28d9);
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-size: 15px;
  font-weight: 900;
  box-shadow: 0 12px 24px rgba(124,58,237,.22);
  margin-bottom: 0 !important;
}

.upload-caption * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

#pdf_upload {
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* Main drop area only */
#pdf_upload .wrap {
  width: 100% !important;
  min-height: 215px !important;
  background: #ffffff !important;
  border: 2px dashed #c4b5fd !important;
  border-radius: 24px !important;
  display: grid !important;
  place-items: center !important;
  text-align: center !important;
  position: relative !important;
  overflow: hidden !important;
}

/* Hide only Gradio placeholder text before upload */
#pdf_upload .wrap > div:first-child,
#pdf_upload .wrap > span,
#pdf_upload .wrap > p {
  display: none !important;
}

/* Custom placeholder */
#pdf_upload .wrap::after {
  content: "☁️\A Drag & drop your PDF file here\A or click to browse";
  white-space: pre-line;
  color: #334155 !important;
  -webkit-text-fill-color: #334155 !important;
  font-size: 16px !important;
  line-height: 1.8 !important;
  font-weight: 700 !important;
  text-align: center !important;
}

/* After file upload: remove fake placeholder */
#pdf_upload:has([data-testid="file"]) .wrap::after,
#pdf_upload:has(.file-preview) .wrap::after {
  content: "" !important;
}

/* Uploaded file card readability */
#pdf_upload [data-testid="file"],
#pdf_upload .file-preview,
#pdf_upload .file-preview *,
#pdf_upload a,
#pdf_upload span,
#pdf_upload p {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  opacity: 1 !important;
}

/* Hide built-in label only */
#pdf_upload .label-wrap,
#pdf_upload label,
#pdf_upload [data-testid="block-label"],
#pdf_upload input[type="file"] {
  display: none !important;
}

/* Engine wrapper */
#engine_radio,
#engine_radio *,
#engine_radio fieldset,
#engine_radio .wrap {
  background: #FFFFFF !important;
  border: none !important;
  box-shadow: none !important;
}

.engine-caption {
  display: inline-flex;
  width: fit-content;
  align-items: center;
  padding: 8px 14px;
  border-radius: 10px;
  background: #7c3aed;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-size: 16px;
  font-weight: 900;
  margin-bottom: 10px;
}


#engine_radio label,
#engine_radio label span {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: #8b5cf6 !important;
  -webkit-text-fill-color: #8b5cf6 !important;
  font-size: 14px !important;
  font-weight: 900 !important;
}

/* Selected option purple */
#engine_radio label:has(input:checked),
#engine_radio label:has(input:checked) * {
  color: #5b21b6 !important;
  -webkit-text-fill-color: #5b21b6 !important;
}

/* Radio circles */
#engine_radio input[type="radio"] {
  display: inline-block !important;
  appearance: auto !important;
  -webkit-appearance: radio !important;
  opacity: 1 !important;
  visibility: visible !important;
  position: static !important;
  width: 14px !important;
  height: 14px !important;
  margin-right: 8px !important;
  accent-color: #5b21b6 !important;
}

#engine_radio .wrap {
  display: flex !important;
  flex-direction: row !important;
  gap: 40px !important;
  align-items: center !important;
}

/* Hide Gradio's built-in title completely */
#engine_radio legend,
#engine_radio .label-wrap,
#engine_radio [data-testid="block-label"] {
  display: none !important;
}


/* EMERGENCY READABILITY FIX */

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6 {
    color: #111827 !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-weight: 800 !important;
}

/* Safe readability fix: Markdown only, not JSON/code internals */
.gradio-container .markdown,
.gradio-container .markdown p,
.gradio-container .markdown li,
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3,
.gradio-container .markdown strong {
  color: #111827 !important;
  opacity: 1 !important;
}

/* Keep code/JSON readable */
.gradio-container pre,
.gradio-container code,
.gradio-container pre *,
.gradio-container code *,
.gradio-container .cm-editor,
.gradio-container .cm-editor * {
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
}

/* ================= FINAL TAB READABILITY PATCH ================= */

/* Normal markdown text, including _Not found._ italic text */
.gradio-container .markdown,
.gradio-container .markdown *,
.gradio-container .prose,
.gradio-container .prose * {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  opacity: 1 !important;
}

/* Markdown italic fallback like _Not found._ */
.gradio-container em,
.gradio-container i {
  color: #475569 !important;
  -webkit-text-fill-color: #475569 !important;
  opacity: 1 !important;
}

/* Inline code like `survey_or_review` */
.gradio-container .markdown code,
.gradio-container p code,
.gradio-container li code {
  background: #f1f5f9 !important;
  color: #5b21b6 !important;
  -webkit-text-fill-color: #5b21b6 !important;
  padding: 2px 6px !important;
  border-radius: 6px !important;
  font-weight: 800 !important;
}

/* Force tab panels to stay white */
.gradio-container .tabitem,
.gradio-container .tabitem > div,
.gradio-container .tabitem .block,
.gradio-container .tabitem .wrap {
  background: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

/* Advanced accordion fix */
.gradio-container .accordion,
.gradio-container .accordion *,
.gradio-container details,
.gradio-container details * {
  background: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

/* Tables in Advanced Analysis */
.gradio-container table,
.gradio-container table *,
.gradio-container th,
.gradio-container td {
  background: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  border-color: #cbd5e1 !important;
}

/* Code blocks inside markdown */
.gradio-container pre,
.gradio-container pre *,
.gradio-container code,
.gradio-container code * {
  background: #f8fafc !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

/* Export JSON/code viewer: make it readable */
.gradio-container .cm-editor,
.gradio-container .cm-editor *,
.gradio-container .cm-scroller,
.gradio-container .cm-content,
.gradio-container .cm-line {
  background: #f8fafc !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

/* Export file boxes */
.gradio-container [data-testid="file"],
.gradio-container [data-testid="file"] *,
.gradio-container .file-preview,
.gradio-container .file-preview * {
  background: #f8fafc !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

/* ================= LAB READINESS RESTORE ================= */

/* Restore dark Tools tile readability */
.lab-board .lab-dark,
.lab-board .lab-dark *,
.lab-board .lab-dark span,
.lab-board .lab-dark b {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  opacity: 1 !important;
}

/* Optional: softer label inside dark card */
.lab-board .lab-dark span {
  color: #c4b5fd !important;
  -webkit-text-fill-color: #c4b5fd !important;
}

/* Keep other lab tiles clean */
.lab-board .lab-green,
.lab-board .lab-green * {
  color: #065f46 !important;
  -webkit-text-fill-color: #065f46 !important;
}

.lab-board .lab-purple,
.lab-board .lab-purple * {
  color: #4c1d95 !important;
  -webkit-text-fill-color: #4c1d95 !important;
}

.lab-board .lab-red,
.lab-board .lab-red * {
  color: #9f1239 !important;
  -webkit-text-fill-color: #9f1239 !important;
}

/* ================= HERO COLOR RESTORE ================= */

.hero,
.hero * {
  color: inherit;
}

.hero .logo-text {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.hero .kicker {
  color: #67e8f9 !important;
  -webkit-text-fill-color: #67e8f9 !important;
}

.hero h1 {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.hero h1 span {
  color: #a78bfa !important;
  -webkit-text-fill-color: #a78bfa !important;
}

.hero p {
  color: rgba(255,255,255,.88) !important;
  -webkit-text-fill-color: rgba(255,255,255,.88) !important;
}

.hero .hero-badge,
.hero .hero-badge * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.hero .logo-mark {
  color: inherit !important;
  -webkit-text-fill-color: inherit !important;
}

/* ================= ASK THE PAPER / RAG FIX ================= */

.rag-panel,
.rag-panel * {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

.rag-question,
.rag-question *,
.rag-question textarea,
.rag-question input {
  background: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  border-color: #cbd5e1 !important;
}

.rag-question textarea::placeholder,
.rag-question input::placeholder {
  color: #64748b !important;
  -webkit-text-fill-color: #64748b !important;
  opacity: 1 !important;
}

.rag-answer {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 18px !important;
  padding: 22px !important;
  box-shadow: 0 8px 24px rgba(15,23,42,.04) !important;
  line-height: 1.65 !important;
}

.rag-answer,
.rag-answer * {
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}

.rag-answer h1,
.rag-answer h2,
.rag-answer h3 {
  margin-top: 18px !important;
  margin-bottom: 10px !important;
  font-weight: 900 !important;
}

.rag-answer ul {
  padding-left: 22px !important;
}

.rag-answer li {
  margin-bottom: 7px !important;
}

.rag-answer blockquote {
  margin: 12px 0 !important;
  padding: 12px 16px !important;
  border-left: 4px solid #7c3aed !important;
  background: #f8fafc !important;
  border-radius: 10px !important;
  color: #334155 !important;
  -webkit-text-fill-color: #334155 !important;
}

.rag-debug,
.rag-debug * {
  background: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
)


with gr.Blocks(title=APP_NAME, theme=theme, css=CSS) as demo:
    state = gr.State({})

    gr.HTML(
        """
        <div class="app-shell">
          <section class="hero">
            <div class="hero-content">
              <div class="logo-row">
                <span class="logo-mark">🧪</span>
                <span class="logo-text">Paper2Lab</span>
              </div>

              <div class="kicker">Research Assistant</div>

              <h1>
                Turn research papers into
                <span>actionable lab plans</span>
              </h1>

              <p>
                Upload a PDF and get a structured paper card, lab starter kit,
                evidence grounding, reproducibility assessment and exportable reports.
              </p>

              <div class="hero-badges">
                <div class="hero-badge">✦ AI-powered extraction</div>
                <div class="hero-badge">□ Evidence grounded</div>
                <div class="hero-badge">☘ Reproducibility ready</div>
              </div>
            </div>
          </section>
        </div>
        """
    )

    with gr.Row(elem_classes=["app-shell", "top-layout"]):
        with gr.Column(scale=4, min_width=300, elem_classes=["control-stack"]):
            gr.HTML("""<div class="panel-title">📤 1. Upload your paper</div>""")
            gr.HTML("""<div class="upload-caption">📄 Drop your PDF here</div>""")

            pdf_input = gr.File(
                label=None,
                show_label=False,
                file_types=[".pdf"],
                type="filepath",
                elem_id="pdf_upload",
                elem_classes=["clean-upload"],
                )
            
            gr.HTML("""<div class="panel-title">⚡ 2. Select analysis engine</div>""")
            gr.HTML("""<div class="engine-caption">⚙️ Analysis engine</div>""")

            refinement_mode = gr.Radio(
                label=None,
                container=False,
                choices=["local", "nemotron"],
                value="nemotron",
                elem_id="engine_radio",
                )

            run_btn = gr.Button(
                "✨ Analyze Paper",
                variant="primary",
                size="lg",
                elem_classes=["clean-button"],
            )

            gr.HTML("""<div class="tip-box">💡 Use a clear-text PDF for best results.</div>""")

        with gr.Column(scale=7, min_width=520, elem_classes=["result-stack"]):
            quick_summary = gr.HTML(
                """
                <div class="summary-card">
                  <div class="card-head">
                    <div><span class="icon">📄</span><b>Quick summary</b></div>
                    <span class="status-pill">Ready</span>
                  </div>
                  <div class="summary-row">
                    <div class="label">Status</div>
                    <div class="value">Upload a PDF and click <b>Analyze Paper</b>.</div>
                  </div>
                </div>
                """
            )

            lab_cards_html = gr.HTML(
                """
                <div class="lab-board">
                  <div class="lab-board-header">
                    <span class="lab-emoji">🧪</span>
                    <span>Lab readiness</span>
                  </div>

                  <div class="lab-grid-2">
                    <div class="lab-tile lab-green">
                      <span>Starter type</span>
                      <b>Waiting</b>
                    </div>

                    <div class="lab-tile lab-dark">
                      <span>Tools</span>
                      <b>Waiting</b>
                    </div>

                    <div class="lab-tile lab-purple">
                      <span>Experiments</span>
                      <b>Waiting</b>
                    </div>

                    <div class="lab-tile lab-red">
                      <span>Risks</span>
                      <b>Waiting</b>
                    </div>
                  </div>
                </div>
                """
            )

    with gr.Row(elem_classes=["app-shell", "tabs-shell"]):
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("📄 Paper Summary"):
                    paper_md = gr.Markdown()

                with gr.Tab("🧪 Lab Starter Kit"):
                    lab_md = gr.Markdown()

                with gr.Tab("🔎 Evidence Viewer"):
                    evidence_md = gr.Markdown()

                with gr.Tab("💬 Ask the Paper"):
                    with gr.Column(elem_classes=["rag-panel"]):
                        rag_question = gr.Textbox(
                            label="Ask a question about the uploaded paper",
                            placeholder="Example: What datasets were used?",
                            lines=2,
                            elem_classes=["rag-question"],
                          )

                        rag_btn = gr.Button("Ask", variant="primary")

                        rag_answer = gr.Markdown(elem_classes=["rag-answer"])

                        rag_json = gr.JSON(
                            label="RAG debug output",
                            elem_classes=["rag-debug"],
                            visible=False,   # better for demo
                          )

                with gr.Tab("⚙️ Advanced Analysis"):
                    with gr.Accordion("Model comparison, selection report, metadata", open=False):
                        advanced_md = gr.Markdown()

                with gr.Tab("⬇️ Export"):
                    final_json = gr.JSON(label="Final paper card")
                    json_file = gr.File(label="Download JSON")
                    md_file = gr.File(label="Download Markdown report")

    run_btn.click(
        fn=analyze_paper,
        inputs=[pdf_input, refinement_mode],
        outputs=[
            state,
            quick_summary,
            paper_md,
            lab_md,
            evidence_md,
            advanced_md,
            lab_cards_html,
            final_json,
            json_file,
            md_file,
        ],
        show_progress=True,
    )

    rag_btn.click(
    fn=ask_paper_question,
    inputs=[state, rag_question],
    outputs=[rag_answer, rag_json],
    show_progress=True,
)

if __name__ == "__main__":
    demo.queue()
    demo.launch()