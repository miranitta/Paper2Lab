# test_nemotron_real.py

import json

from paper2lab.data.pdf_loader import extract_pdf
from paper2lab.inference.paper_card import build_paper_card
from paper2lab.inference.roadmap import build_reproduction_roadmap
from paper2lab.evaluation.reproducibility import reproducibility_report
from paper2lab.inference.visual_explainer import explain_figures_and_tables
from paper2lab.inference.nemotron_refiner import refine_with_nemotron

pdf_path = "Data/papers/train/Education intervention.pdf"

extracted = extract_pdf(pdf_path)
card = build_paper_card(extracted)

roadmap = build_reproduction_roadmap(extracted, card)
figures_tables = explain_figures_and_tables(extracted)
repro_score = reproducibility_report(extracted, card)

card["reproduction_roadmap"] = roadmap
card["figures_and_tables"] = figures_tables
card["reproducibility_score"] = repro_score

pack = card["llm_evidence_pack"]
pack["candidate_paper_card"] = {
    k: v for k, v in card.items()
    if k != "llm_evidence_pack"
}

result = refine_with_nemotron(pack)

print(json.dumps(result, indent=2, ensure_ascii=False))