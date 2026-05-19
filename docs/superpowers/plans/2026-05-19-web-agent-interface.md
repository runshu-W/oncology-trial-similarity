# Web Agent Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local browser interface that accepts a pasted or uploaded trial JSON and runs the existing oncology trial similarity pipeline.

**Architecture:** Add a small FastAPI app under `web_agent/` that imports the existing pipeline module and exposes `/api/health` and `/api/search`. Serve one static HTML/CSS/JS page that submits JSON, renders query metadata and ranked candidates, and exposes raw JSON/Markdown downloads. Keep the pipeline file unchanged except for reuse through imports.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, stdlib `unittest`, plain HTML/CSS/JavaScript.

---

### Task 1: Add Backend Tests

**Files:**
- Create: `tests/test_web_agent.py`
- Create: `web_agent/__init__.py`
- Create: `web_agent/app.py`

- [ ] **Step 1: Write failing tests for health, invalid JSON, and mocked search**

Create `tests/test_web_agent.py` with tests that import `web_agent.app`, monkeypatch the search runner, and verify API behavior without loading the full ClinicalBERT index.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_web_agent -v`
Expected: fail because `web_agent.app` does not exist.

### Task 2: Implement FastAPI Backend

**Files:**
- Modify: `web_agent/app.py`
- Modify: `web_agent/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add minimal FastAPI app**

Implement `/api/health`, `/api/search`, static file serving, JSON validation, temporary query-file creation, and calls to existing `search()` plus `render_markdown_report()`.

- [ ] **Step 2: Run backend tests**

Run: `.venv/bin/python -m unittest tests.test_web_agent -v`
Expected: pass.

### Task 3: Add Frontend

**Files:**
- Create: `web_agent/static/index.html`
- Modify: `web_agent/app.py`

- [ ] **Step 1: Build single-page UI**

Create upload/paste input controls, run button, status area, summary panel, candidate cards, red flag rendering, and raw JSON/Markdown download buttons.

- [ ] **Step 2: Verify static page route**

Run: `.venv/bin/python -m unittest tests.test_web_agent -v`
Expected: pass, including `GET /`.

### Task 4: Document Running the Agent

**Files:**
- Modify: `README.md`
- Create: `web_agent/README.md`

- [ ] **Step 1: Add local run instructions**

Document dependency install, startup command, default index path, and limitations.

- [ ] **Step 2: Verify docs mention no final borrowing decision**

Run: `rg -n "not a final|not final|expert" README.md web_agent/README.md`
Expected: at least one clear warning.

### Task 5: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run syntax and unit tests**

Run: `.venv/bin/python -m py_compile docs/oncology_trial_similarity_pipeline.py web_agent/app.py scripts/build_manual_evaluation_template.py && .venv/bin/python -m unittest tests.test_web_agent -v`
Expected: exit code 0.

- [ ] **Step 2: Run one mocked API request through TestClient**

Covered by `tests/test_web_agent.py`; confirm output includes `result`, `markdown_report`, and candidate data.
