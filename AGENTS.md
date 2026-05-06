# Repository Guidelines

## Project Structure & Module Organization
Core backend modules live at the repo root: `main.py` (FastAPI app), `rag_core.py` (RAG pipeline), `pdf_processor.py` (document parsing), and `celery_worker.py` (async tasks). Shared subsystems are organized as packages like `delegator/` (bot orchestration) and `storage/` (storage backends). UI assets live in `frontend/` (chat app) and `landing_page/` (marketing + API docs). Tests are in `tests/` plus root-level `test_*.py`. Data and artifacts are stored under `datasets/`, `documents/`, `finqa_test_data/`, `vector_stores/`, and `user_data/`. Historical or scratch code is kept in `obsolete/`, `backup_refactor/`, and `BACKUP/` and should not be modified for new work.

## Build, Test, and Development Commands
Install dependencies from `requirements.txt`, then use the commands below for local workflows:

```bash
pip install -r requirements.txt
python main.py
uvicorn main:app --host 0.0.0.0 --port 5000
celery -A celery_worker.celery_app worker --loglevel=info
python delegator.py
pytest tests/
pytest tests/test_numeric_extraction.py -v
python build_vector_store.py --corpus data.jsonl --output vector_stores/output
```

`celery` requires Redis running. The UI can be served by the API (`/` and `/ask` pages).

## Coding Style & Naming Conventions
Use 4-space indentation in Python and JavaScript. Follow Python naming: `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. There is no enforced formatter, so match the surrounding file’s style and keep changes minimal and focused.

## Testing Guidelines
Tests use `pytest` and are primarily located in `tests/` with some `test_*.py` at the repo root. Prefer targeted runs (e.g., `pytest tests/test_numeric_extraction.py -v`) before running the full suite. Integration tests may require configured services (Groq/OpenAI keys, Firebase/Firestore, Redis, Neo4j).

## Commit & Pull Request Guidelines
The Git history currently contains a single commit (`"Initial commit: …"`), so no strict convention exists. Use concise, imperative summaries and include a short scope when helpful (e.g., `RAG: fix numeric extraction edge case`). PRs should include a short summary, tests run, and any required services or environment variables; include screenshots for changes in `frontend/` or `landing_page/`.

## Security & Configuration Tips
Secrets are loaded from `.env`; do not commit them. Document any new required variables in `PROJECT_SPECIFICATIONS.md` and keep sample values out of the repo.

## AI for Bharat Theme 11 Context
This repo contains a Phase 2 hackathon prototype for AI for Bharat Theme 11: converting court judgment PDFs into evidence-backed, human-verified action plans and a trusted dashboard. Before modifying this workflow, read `THEME11_PHASE2_HANDOFF.md` and `AI_FOR_BHARAT_ROUND1_THEME11_SUBMISSION.md`.

Primary Theme 11 files:

- Backend API/workflow: `judgment_workflow/api.py`, `judgment_workflow/pipeline.py`, `judgment_workflow/repository.py`
- Judgment intelligence: `rag/judgment/extractor.py`, `rag/judgment/action_plan.py`, `rag/judgment/retrieval.py`, `rag/judgment/types.py`
- Source proof/metrics/duplicates: `judgment_workflow/pdf_highlights.py`, `judgment_workflow/metrics.py`, `judgment_workflow/document_layers.py`
- Frontend: `frontend/judgments.html`, `frontend/judgments.css`, `frontend/judgments.js`
- Tests: `tests/test_judgment_core.py`, `tests/test_judgment_workflow.py`

Current Theme 11 demo/UI expectations:

- Preserve the polished Operations Console visual direction in `/judgments`.
- Do not reintroduce the old left navigation rail or top-right clock/status clutter unless explicitly requested.
- Keep user-facing labels human-readable (`Pending Review`, not `pending_review`).
- Keep the review drawer action-first: proposed action plan should be easy to see before deeper extraction details.
- Extraction rows should remain provenance-based with source/evidence chips when data exists.
- Review controls should support field status/reason capture for statuses such as approved, edited, ambiguous, wrong, and empty.
- Queue risk indicators should remain compact severity-colored chips, not long raw risk strings.
- Avoid showing noisy internal fields such as legal phrases in the compact reviewer drawer unless the user explicitly wants them.

Scale/evaluation API:

- `POST /judgments/evaluate` accepts a PDF as multipart form-data, runs the same judgment pipeline as the UI, stores a review record, and returns a clean `judgment-evaluation-v1` schema for accuracy testing.
- Form fields: `file` (required), `user_id` (optional, defaults to `evaluation`), `include_full_record` (optional bool), `llm_enabled` (optional bool), and `metadata_json` (optional JSON object string).
- Use `llm_enabled=false` for deterministic regression baselines; use `llm_enabled=true` to compare the optional LLM enrichment path.
- The endpoint does not compute ground-truth accuracy by itself. External scripts should compare `extraction.fields`, `extraction.directions`, `action_plan.items`, `quality.metrics`, and `quality.risk_flags` against human-labeled expected outputs.
- Do not fork a separate extraction path for evaluation unless absolutely necessary; keep `/judgments/evaluate`, `/judgments/upload`, and the reviewer UI backed by the same `process_judgment_file()` pipeline.

For local demo, the app is normally served at `http://127.0.0.1:5000/judgments` with:

```powershell
$env:STORAGE_BACKEND='sqlite'
$env:SQLITE_DB_PATH='user_data\local_demo\artha.db'
$env:JUDGMENT_SYNC_PROCESSING='1'
$env:DISABLE_RAG_STARTUP='1'
$env:JUDGMENT_DEMO_AUTH_BYPASS='1'
$env:CANVAS_APP_ID='theme11-local'
$env:APP_SESSION_SECRET='dev-session-secret-change-me'
$env:PROGRESS_HMAC_SECRET='dev-progress-secret-change-me'
$env:JUDGMENT_DATA_ROOT='user_data\judgments'
python -m uvicorn main:app --host 127.0.0.1 --port 5000
```

`JUDGMENT_DEMO_AUTH_BYPASS=1` is for local demo only. Do not describe auth/RBAC as production-ready.

Before claiming Theme 11 work is complete, run:

```powershell
python -m pytest tests/test_judgment_core.py tests/test_judgment_workflow.py -q
node --check frontend\judgments.js
python -m py_compile judgment_workflow\api.py judgment_workflow\pipeline.py judgment_workflow\repository.py judgment_workflow\metrics.py judgment_workflow\document_layers.py judgment_workflow\pdf_highlights.py judgment_workflow\serialization.py rag\judgment\retrieval.py rag\judgment\service.py rag\judgment\types.py
```

Useful local evaluation smoke test:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/judgments/evaluate" `
  -F "user_id=evaluation-smoke" `
  -F "include_full_record=false" `
  -F "llm_enabled=false" `
  -F "file=@C:\path\to\judgment.pdf;type=application/pdf"
```

The expected response should include `schema_version: judgment-evaluation-v1`, a `record_id`, `extraction.fields`, `action_plan.items`, `quality.metrics`, and `artifacts.highlighted_pdf_url`.

Honest positioning: this is a working Phase 2 prototype aligned with the idea submission. Do not overstate production readiness. Real CCMS integration, production auth/RBAC, legal-grade extraction accuracy, perfect OCR, and perfect PDF highlights are not guaranteed. The sandbox CCMS adapter, human review workflow, source evidence, audit trail, duplicate resolution, metrics, and trusted dashboard are implemented for prototype/demo use.
