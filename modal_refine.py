from __future__ import annotations

import modal
from fastapi import Request

app = modal.App("paper2lab-nemotron")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "fastapi")
    .add_local_dir("src/paper2lab", remote_path="/root/paper2lab")
)

secret = modal.Secret.from_name("nvidia-api-key")


@app.function(
    image=image,
    secrets=[secret],
    timeout=300,
)
@modal.fastapi_endpoint(method="POST")
async def refine_remote(request: Request):
    from paper2lab.inference.nemotron_refiner import refine_with_nemotron

    body = await request.json()

    llm_evidence_pack = body.get("llm_evidence_pack")
    model = body.get("model", "nvidia/nemotron-3-nano-30b-a3b")
    return_comparison = body.get("return_comparison", True)

    if not llm_evidence_pack:
        return {
            "status": "error",
            "error": "Missing llm_evidence_pack",
        }

    try:
        result = refine_with_nemotron(
            llm_evidence_pack=llm_evidence_pack,
            model=model,
            return_comparison=return_comparison,
        )
        return {
            "status": "ok",
            "result": result,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


@app.local_entrypoint()
def main():
    print("Deploy this app with: modal deploy modal_refine.py")