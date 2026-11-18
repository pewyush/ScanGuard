"""ScanGuard comparison dashboard.

Serves the metrics from the latest comparison run, the generated figures, and
live anomaly scores from both models. Run with::

    uv run scanguard serve
"""

import html
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from scanguard.serving.predictor import AnomalyPredictor, PaDiMAnomalyPredictor

CATEGORY = os.getenv("SCANGUARD_CATEGORY", "can")
RESULTS_PATH = Path(os.getenv("SCANGUARD_RESULTS", f"results/{CATEGORY}_results.json"))
CHECKPOINT_PATH = Path(os.getenv("SCANGUARD_CHECKPOINT", f"checkpoints/{CATEGORY}_best.pt"))
PADIM_PATH = Path(os.getenv("SCANGUARD_PADIM", f"checkpoints/{CATEGORY}_padim.npz"))
FIGURES_DIR = Path(os.getenv("SCANGUARD_FIGURES", "results/figures"))

METRIC_ROWS = [
    ("Image AUROC", "image_auroc"),
    ("Image AUPR", "image_aupr"),
    ("Image precision", "precision"),
    ("Image recall", "recall"),
    ("Image F1", "f1"),
    ("Pixel AUROC", "pixel_auroc"),
    ("AU-PRO (0.30)", "au_pro_0.30"),
    ("AU-PRO (0.05)", "au_pro_0.05"),
]


class PredictRequest(BaseModel):
    image_base64: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.results = None
    app.state.ae = None
    app.state.padim = None
    app.state.errors = {}

    if RESULTS_PATH.exists():
        try:
            app.state.results = json.loads(RESULTS_PATH.read_text())
        except Exception as exc:
            app.state.errors["results"] = f"{type(exc).__name__}: {exc}"

    if CHECKPOINT_PATH.exists():
        try:
            app.state.ae = AnomalyPredictor.from_checkpoint(CHECKPOINT_PATH)
        except Exception as exc:
            app.state.errors["autoencoder"] = f"{type(exc).__name__}: {exc}"
    else:
        app.state.errors["autoencoder"] = f"checkpoint not found: {CHECKPOINT_PATH}"

    if PADIM_PATH.exists():
        try:
            app.state.padim = PaDiMAnomalyPredictor.from_file(PADIM_PATH)
        except Exception as exc:
            app.state.errors["padim"] = f"{type(exc).__name__}: {exc}"
    else:
        app.state.errors["padim"] = f"PaDiM statistics not found: {PADIM_PATH}"

    yield


app = FastAPI(title="ScanGuard", lifespan=lifespan)


def _format_metrics(results: Optional[dict]) -> str:
    if not results:
        return (
            '<p class="warn">No comparison results yet. Run '
            "<code>uv run scanguard train</code> for a quick check, or "
            "<code>uv run scanguard train --full</code> for the full run.</p>"
        )

    meta = results["meta"]
    ae = results["autoencoder"]["metrics"]
    padim = results["padim"]["metrics"]
    mode = "quick slice" if meta["quick"] else "full data"
    meta_line = (
        f"Category <b>{html.escape(meta['category'])}</b> · {mode} · "
        f"input {meta['img_size']}×{meta['img_size']} · device {html.escape(meta['device'])} · "
        f"train/val/test {meta['n_train']}/{meta['n_val']}/{meta['n_test']} · "
        f"{meta['n_test_anomaly']} anomalous test images"
    )

    rows = []
    for label, key in METRIC_ROWS:
        rows.append(
            f"<tr><td>{label}</td><td>{ae[key]:.4f}</td><td>{padim[key]:.4f}</td></tr>"
        )
    rows.append(
        f"<tr class=\"muted\"><td>Operating threshold</td>"
        f"<td>{results['autoencoder']['threshold']:.4f}</td>"
        f"<td>{results['padim']['threshold']:.4f}</td></tr>"
    )

    return (
        f'<p class="meta">{meta_line}</p>'
        '<table><thead><tr><th>Metric</th><th>Autoencoder</th>'
        '<th>PaDiM</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
        '<p class="hint">Thresholds are the 99th percentile of defect-free '
        "validation scores. AU-PRO is per-region overlap over FPR ∈ [0, 0.30] "
        "and [0, 0.05].</p>"
    )


def _format_gallery() -> str:
    if not FIGURES_DIR.is_dir():
        return '<p class="hint">No figures generated yet.</p>'

    figures = sorted(FIGURES_DIR.glob("*.png"))
    if not figures:
        return '<p class="hint">No figures generated yet.</p>'

    items = []
    for figure in figures:
        name = html.escape(figure.name)
        items.append(
            f'<figure><a href="/figures/{quote(figure.name)}" target="_blank">'
            f'<img src="/figures/{quote(figure.name)}" alt="{name}"></a>'
            f"<figcaption>{name}</figcaption></figure>"
        )
    return '<div class="gallery">' + "".join(items) + "</div>"


def _render_index(state) -> str:
    page = INDEX_HTML
    page = page.replace("<!--METRICS-->", _format_metrics(state.results))
    page = page.replace("<!--GALLERY-->", _format_gallery())

    issues = []
    for key in ("autoencoder", "padim"):
        if state.errors.get(key):
            issues.append(f'<li><b>{key}</b>: {html.escape(state.errors[key])}</li>')
    if issues:
        page = page.replace(
            "<!--WARNINGS-->", '<ul class="warn">' + "".join(issues) + "</ul>"
        )
    else:
        page = page.replace("<!--WARNINGS-->", "")
    return page


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScanGuard — autoencoder vs PaDiM</title>
<style>
  :root { color-scheme: light; }
  body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0;
         background: #f5f6f8; color: #1b1d21; }
  header { background: #15181e; color: #fff; padding: 1.4rem 2rem 1.2rem; }
  header h1 { margin: 0; font-size: 1.5rem; letter-spacing: .02em; }
  header p { margin: .25rem 0 0; color: #b8c0cc; }
  main { max-width: 1080px; margin: 1.5rem auto 3rem; padding: 0 1.25rem; }
  .panel { background: #fff; border: 1px solid #e0e3e8; border-radius: 12px;
           padding: 1.2rem 1.3rem; margin-bottom: 1.2rem; box-shadow: 0 1px 2px rgba(0,0,0,.03); }
  .panel h2 { margin: 0 0 .9rem; font-size: 1.05rem; }
  .meta { color: #46505f; font-size: .92rem; margin: 0 0 .8rem; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { padding: .45rem .6rem; border-bottom: 1px solid #eceef2; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  thead th { color: #5b6575; font-weight: 600; font-size: .9rem; }
  tr.muted td { color: #7b8494; }
  .hint { color: #6c7686; font-size: .85rem; margin: .8rem 0 0; }
  .warn { color: #a3341f; background: #fdf0ed; border: 1px solid #f3c9c0;
          border-radius: 8px; padding: .6rem .8rem; font-size: .9rem; }
  code { background: #eef0f4; padding: .1rem .35rem; border-radius: 4px; }
  .upload { display: flex; gap: .8rem; align-items: center; flex-wrap: wrap; }
  button { padding: .55rem 1.1rem; border: 0; border-radius: 8px; background: #15181e;
           color: #fff; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .45; cursor: default; }
  #status { color: #5b6575; font-size: .9rem; }
  .results { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; }
  .card { border: 1px solid #e0e3e8; border-radius: 10px; padding: .8rem; }
  .card h3 { margin: 0 0 .35rem; font-size: .95rem; }
  .score { font-size: 1.5rem; font-variant-numeric: tabular-nums; }
  .verdict { font-size: .8rem; font-weight: 700; letter-spacing: .04em; margin: .15rem 0 .5rem; }
  .verdict.ok { color: #1f7a48; }
  .verdict.defect { color: #b02a1b; }
  .card img { width: 100%; border-radius: 6px; display: block; background: #f0f1f4; }
  .card .err { color: #a3341f; font-size: .85rem; }
  .gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }
  figure { margin: 0; }
  figure img { width: 100%; border: 1px solid #e0e3e8; border-radius: 8px; display: block; }
  figcaption { color: #6c7686; font-size: .8rem; margin-top: .3rem; }
  footer { color: #7b8494; font-size: .82rem; text-align: center; padding-bottom: 2rem; }
  @media (max-width: 760px) { .results { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>ScanGuard</h1>
  <p>Unsupervised visual anomaly detection — reconstruction baseline vs PaDiM</p>
</header>
<main>
  <!--WARNINGS-->
  <section class="panel">
    <h2>Comparison results</h2>
    <!--METRICS-->
  </section>

  <section class="panel">
    <h2>Inspect an image</h2>
    <div class="upload">
      <input id="file" type="file" accept="image/*">
      <button id="submit" disabled>Inspect</button>
      <span id="status">Choose a product image.</span>
    </div>
    <div class="results">
      <div class="card">
        <h3>Input</h3>
        <div class="score">&nbsp;</div>
        <div class="verdict">&nbsp;</div>
        <img id="preview" alt="input preview" hidden>
      </div>
      <div class="card">
        <h3>Autoencoder</h3>
        <div class="score" id="ae-score">—</div>
        <div class="verdict" id="ae-verdict"></div>
        <img id="ae-heatmap" alt="autoencoder anomaly map" hidden>
        <div class="err" id="ae-error"></div>
      </div>
      <div class="card">
        <h3>PaDiM</h3>
        <div class="score" id="padim-score">—</div>
        <div class="verdict" id="padim-verdict"></div>
        <img id="padim-heatmap" alt="PaDiM anomaly map" hidden>
        <div class="err" id="padim-error"></div>
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Result figures</h2>
    <!--GALLERY-->
  </section>
</main>
<footer>ScanGuard · MVTec AD 2 · academic project</footer>
<script>
const fileInput = document.getElementById("file");
const submit = document.getElementById("submit");
const status = document.getElementById("status");
const preview = document.getElementById("preview");

function resetResult(prefix) {
  document.getElementById(prefix + "-score").textContent = "—";
  const verdict = document.getElementById(prefix + "-verdict");
  verdict.textContent = "";
  verdict.className = "verdict";
  const heatmap = document.getElementById(prefix + "-heatmap");
  heatmap.hidden = true;
  heatmap.removeAttribute("src");
  document.getElementById(prefix + "-error").textContent = "";
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  submit.disabled = !file;
  resetResult("ae");
  resetResult("padim");
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  status.textContent = file.name;
});

function showResult(prefix, result) {
  if (!result || result.error) {
    document.getElementById(prefix + "-error").textContent =
      result ? result.error : "Not available.";
    return;
  }
  document.getElementById(prefix + "-score").textContent = result.score.toFixed(4);
  const verdict = document.getElementById(prefix + "-verdict");
  verdict.textContent = result.verdict;
  verdict.className = "verdict " + (result.verdict === "OK" ? "ok" : "defect");
  const heatmap = document.getElementById(prefix + "-heatmap");
  heatmap.src = result.heatmap;
  heatmap.hidden = false;
}

submit.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  submit.disabled = true;
  status.textContent = "Scoring…";
  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: dataUrl })
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || response.statusText);
    }
    const result = await response.json();
    showResult("ae", result.autoencoder);
    showResult("padim", result.padim);
    status.textContent = "Done.";
  } catch (error) {
    status.textContent = "Error: " + error.message;
  } finally {
    submit.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render_index(app.state)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "autoencoder_loaded": app.state.ae is not None,
        "padim_loaded": app.state.padim is not None,
        "results_loaded": app.state.results is not None,
        "results": str(RESULTS_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "padim": str(PADIM_PATH),
        "errors": app.state.errors,
    }


@app.get("/api/results")
def api_results():
    if app.state.results is None:
        raise HTTPException(status_code=404, detail="No results available")
    return JSONResponse(app.state.results)


@app.get("/figures/{name}")
def figure(name: str):
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=404, detail="Not found")
    path = FIGURES_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


def _decode_image(data: str) -> Image.Image:
    import base64
    import io

    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _thresholds(results: Optional[dict]) -> dict:
    if not results:
        return {"autoencoder": None, "padim": None}
    return {
        "autoencoder": results["autoencoder"]["threshold"],
        "padim": results["padim"]["threshold"],
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    image = _decode_image(request.image_base64)
    thresholds = _thresholds(app.state.results)
    response = {}

    for key, predictor in (("autoencoder", app.state.ae), ("padim", app.state.padim)):
        if predictor is None:
            response[key] = {
                "error": app.state.errors.get(key, "Model unavailable."),
                "score": None,
                "heatmap": None,
                "verdict": None,
            }
            continue
        prediction = predictor.predict(image)
        threshold = thresholds.get(key)
        verdict = None
        if threshold is not None:
            verdict = "DEFECT" if prediction.score >= threshold else "OK"
        response[key] = {
            "score": prediction.score,
            "threshold": threshold,
            "verdict": verdict,
            "heatmap": prediction.heatmap_data_uri,
            "error": None,
        }
    return response


def run() -> None:
    """Console entry point for ``scanguard-serve``."""
    uvicorn.run(
        "scanguard.serving.app:app",
        host=os.getenv("SCANGUARD_HOST", "127.0.0.1"),
        port=int(os.getenv("SCANGUARD_PORT", "8000")),
    )


if __name__ == "__main__":
    run()
