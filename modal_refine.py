from __future__ import annotations

import os
import requests
import modal
from fastapi import Request

app = modal.App("paper2lab-nemotron")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "fastapi")
)

secret = modal.Secret.from_name("nvidia-api-key")

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


@app.function(image=image, secrets=[secret], timeout=300)
@modal.fastapi_endpoint(method="POST")
async def refine_remote(request: Request):
    body = await request.json()

    prompt = body.get("prompt")
    model = body.get("model", DEFAULT_MODEL)

    if not prompt:
        return {"status": "error", "error": "Missing prompt"}

    api_key = os.environ["NVIDIA_API_KEY"]

    response = requests.post(
        NVIDIA_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise scientific JSON refiner. Return only valid JSON. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "top_p": 0.7,
            "max_tokens": 8192,
        },
        timeout=180,
    )

    if not response.ok:
        return {"status": "error", "error": response.text[:1000]}

    data = response.json()
    return {
        "status": "ok",
        "content": data["choices"][0]["message"]["content"],
    }