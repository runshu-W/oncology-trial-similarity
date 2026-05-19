from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PIPELINE_PATH = PROJECT_ROOT / "docs" / "oncology_trial_similarity_pipeline.py"
STATIC_DIR = PROJECT_ROOT / "web_agent" / "static"
DEFAULT_INDEX_DIR = WORKSPACE_ROOT / "artifacts" / "oncology_trial_similarity_clinicalbert"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("oncology_trial_similarity_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline module from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline_module()


app = FastAPI(title="Oncology Trial Similarity Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def parse_query_payload(json_text: str | None, upload: UploadFile | None) -> dict[str, Any]:
    if upload is not None and upload.filename:
        raw = upload.file.read().decode("utf-8")
    else:
        raw = json_text or ""
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Provide a trial JSON file or paste JSON text.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Trial JSON must be an object.")
    return payload


def run_pipeline_search(
    query_payload: dict[str, Any],
    index_dir: Path,
    top_k: int,
    rerank_top_n: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="trial-query-") as tmp_dir:
        query_path = Path(tmp_dir) / "query.json"
        query_path.write_text(json.dumps(query_payload, ensure_ascii=False), encoding="utf-8")
        return pipeline.search(
            query_json=query_path,
            index_dir=index_dir,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "index_dir": str(DEFAULT_INDEX_DIR),
        "index_available": (DEFAULT_INDEX_DIR / "trial_summaries.jsonl").exists()
        and (DEFAULT_INDEX_DIR / "trial_embeddings.npz").exists(),
        "default_top_k": 10,
        "default_rerank_top_n": 100,
        "default_rerank": True,
    }


@app.post("/api/search")
def search(
    json_text: str | None = Form(default=None),
    upload: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    query_payload = parse_query_payload(json_text, upload)
    try:
        result = run_pipeline_search(
            query_payload=query_payload,
            index_dir=DEFAULT_INDEX_DIR,
            top_k=10,
            rerank_top_n=100,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "result": result,
        "markdown_report": pipeline.render_markdown_report(result),
    }
