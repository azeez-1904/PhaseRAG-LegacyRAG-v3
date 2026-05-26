#!/usr/bin/env python3
"""
web_ui/app.py — PhaseRAG v3 Web UI (FastAPI + SSE streaming)

Minimal interface: text query, document upload, streaming response,
live VRAM/tok/s stats. No JavaScript frameworks, plain HTML + fetch API.
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="PhaseRAG v3")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
UPLOAD_DIR = Path("/tmp/legacyrag_v3_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_vram_stats() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        result = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                result.append({
                    "gpu": int(parts[0]),
                    "used_mb": float(parts[1]),
                    "total_mb": float(parts[2]),
                    "util_pct": float(parts[3]) if len(parts) > 3 else None,
                })
        return result
    except Exception:
        return []


async def stream_inference(prompt: str) -> AsyncGenerator[str, None]:
    """Stream tokens from llama-server via SSE."""
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": 512,
        "temperature": 0.1,
        "stop": ["</s>", "<|end|>"],
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    tokens_generated = 0

    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                token = data.get("content", "")
                if token:
                    tokens_generated += 1
                    elapsed = time.perf_counter() - t_start
                    tok_s = tokens_generated / elapsed if elapsed > 0 else 0

                    vram = get_vram_stats()
                    vram_str = " | ".join(
                        f"GPU{g['gpu']}: {g['used_mb']:.0f}/{g['total_mb']:.0f}MB"
                        for g in vram
                    ) if vram else "N/A"

                    event = {
                        "token": token,
                        "tok_s": round(tok_s, 2),
                        "tokens": tokens_generated,
                        "elapsed_s": round(elapsed, 1),
                        "vram": vram_str,
                        "stop": data.get("stop", False),
                    }
                    yield f"data: {json.dumps(event)}\n\n"

                    if data.get("stop"):
                        break
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.post("/query")
async def query_endpoint(
    question: str = Form(...),
    document: UploadFile | None = File(None),
):
    context = ""
    if document and document.filename:
        content = await document.read()
        try:
            context = content.decode("utf-8", errors="replace")
        except Exception:
            context = ""
        context = context[:4000]

    if context:
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    else:
        prompt = f"Question: {question}\n\nAnswer:"

    return StreamingResponse(
        stream_inference(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats")
async def stats():
    vram = get_vram_stats()
    try:
        with urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2) as r:
            server_status = "online" if r.status == 200 else "degraded"
    except Exception:
        server_status = "offline"
    return {"vram": vram, "server_status": server_status}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
