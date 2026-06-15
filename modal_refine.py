from __future__ import annotations

import modal

app = modal.App("paper2lab-nemotron")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests")
    .add_local_dir("src/paper2lab", remote_path="/root/paper2lab")
)

secret = modal.Secret.from_name("nvidia-api-key")


@app.function(
    image=image,
    secrets=[secret],
    timeout=180,
)
def refine_remote(
    llm_evidence_pack: dict,
    model: str = "nvidia/nemotron-3-nano-30b-a3b",
    return_comparison: bool = True,
) -> dict:
    from paper2lab.inference.nemotron_refiner import refine_with_nemotron

    return refine_with_nemotron(
        llm_evidence_pack=llm_evidence_pack,
        model=model,
        return_comparison=return_comparison,
    )


@app.local_entrypoint()
def main():
    sample_pack = {
        "candidate_paper_card": {
            "title": "Attention Is All You Need",
            "field": "Natural Language Processing",
            "paper_type": "machine_learning",
            "research_question": "The paper proposes the Transformer architecture.",
            "contributions": [
                "The paper proposes a Transformer architecture based on attention."
            ],
            "methodology": [
                "The model uses multi-head self-attention and feed-forward layers."
            ],
            "datasets_or_data_sources": [
                "WMT 2014 English-German dataset",
                "RNN",
                "BerkleyParser"
            ],
            "models_or_methods": [
                "Transformer",
                "multi-head attention"
            ],
            "metrics_or_measurements": [
                "BLEU"
            ],
            "key_findings": [],
            "limitations": [],
            "missing_reproducibility_info": [
                "random seed is not specified"
            ],
            "metadata": {},
            "source_pdf": "attention is all you need.pdf",
            "annotation_version": "v1.0",
        },
        "section_previews": [
            {
                "title": "Training Data and Batching",
                "role_hint": "methodology",
                "page_start": 7,
                "page_end": 7,
                "preview": "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences."
            }
        ],
        "captions": [],
        "tables": [],
        "metadata": {}
    }

    result = refine_remote.remote(sample_pack)
    print(result)