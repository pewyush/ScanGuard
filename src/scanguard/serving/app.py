"""Minimal FastAPI demo for the ScanGuard anomaly detector.

Run with::

    PYTHONPATH=src uvicorn scanguard.serving.app:app --reload

Point it at a checkpoint with ``SCANGUARD_CHECKPOINT``; the default is
``checkpoints/can_best.pt``.
"""

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from scanguard.serving.predictor import AnomalyPredictor

DEFAULT_CHECKPOINT = Path("checkpoints/can_best.pt")


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    score: float
    heatmap: str


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScanGuard</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 3rem auto; padding: 0 1rem; color: #1b1b1b; }
  h1 { margin-bottom: .25rem; }
  p.lead { color: #555; margin-top: 0; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: start; }
  .card { flex: 1 1 260px; border: 1px solid #ddd; border-radius: 10px; padding: 1rem; }
  .card img { width: 100%; border-radius: 6px; display: block; }
  input[type=file] { margin: .5rem 0; }
  button { padding: .55rem 1rem; border: 0; border-radius: 8px; background: #1b1b1b; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .score { font-size: 1.6rem; font-variant-numeric: tabular-nums; }
  .muted { color: #777; font-size: .9rem; }
</style>
</head>
<body>
<h1>ScanGuard</h1>
<p class="lead">Upload a product image to score reconstruction error and view its anomaly map.</p>
<div class="row">
  <div class="card">
    <input id="file" type="file" accept="image/*">
    <button id="submit" disabled>Inspect</button>
    <p class="muted" id="status">Choose an image.</p>
    <img id="preview" alt="" hidden>
  </div>
  <div class="card">
    <div class="muted">Anomaly score</div>
    <div class="score" id="score">—</div>
    <img id="heatmap" alt="" hidden>
  </div>
</div>
<script>
const file = document.getElementById("file");
const submit = document.getElementById("submit");
const status = document.getElementById("status");
const preview = document.getElementById("preview");
const heatmap = document.getElementById("heatmap");
const score = document.getElementById("score");

file.addEventListener("change", () => {
  const f = file.files[0];
  submit.disabled = !f;
  if (!f) return;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false;
  status.textContent = f.name;
  score.textContent = "—";
  heatmap.hidden = true;
});

submit.addEventListener("click", async () => {
  const f = file.files[0];
  if (!f) return;
  submit.disabled = true;
  status.textContent = "Scoring…";
  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(f);
    });
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: dataUrl })
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const result = await response.json();
    score.textContent = result.score.toFixed(4);
    heatmap.src = result.heatmap;
    heatmap.hidden = false;
    status.textContent = "Done.";
  } catch (err) {
    status.textContent = "Error: " + err.message;
  } finally {
    submit.disabled = false;
  }
});
</script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(os.getenv("SCANGUARD_CHECKPOINT", DEFAULT_CHECKPOINT))
    app.state.checkpoint = checkpoint
    app.state.predictor: Optional[AnomalyPredictor] = None
    app.state.load_error: Optional[str] = None
    if checkpoint.exists():
        try:
            app.state.predictor = AnomalyPredictor.from_checkpoint(checkpoint)
        except Exception as exc:  # surface the reason through /health
            app.state.load_error = f"{type(exc).__name__}: {exc}"
    else:
        app.state.load_error = f"checkpoint not found: {checkpoint}"
    yield


app = FastAPI(title="ScanGuard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if app.state.predictor else "unavailable",
        "model_loaded": app.state.predictor is not None,
        "checkpoint": str(app.state.checkpoint),
        "error": app.state.load_error,
    }


def _decode_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail=app.state.load_error or "Model unavailable")

    prediction = predictor.predict(_decode_image(request.image_base64))
    return PredictResponse(score=prediction.score, heatmap=prediction.heatmap_data_uri)


def run() -> None:
    """Console entry point for ``scanguard-serve``."""
    uvicorn.run(
        "scanguard.serving.app:app",
        host=os.getenv("SCANGUARD_HOST", "127.0.0.1"),
        port=int(os.getenv("SCANGUARD_PORT", "8000")),
    )


if __name__ == "__main__":
    run()
