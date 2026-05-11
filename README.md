# AI For Bharat Theme 11 Prototype

Court-judgment-to-action-plan prototype for the AI for Bharat Phase 2 round.

The demo flow is:

1. Upload a judgment PDF, fetch a CCMS case, or load the synthetic demo seed.
2. The system extracts case metadata, parties, departments, disposition, directions, deadlines, and source evidence.
3. A reviewer verifies or edits extracted fields and action items.
4. Approved/edited/completed records move into the trusted dashboard with filters and CSV/JSON export.
5. Audit history and highlighted PDF source proof stay attached to every record.

## Quick Start

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
$env:JUDGMENT_SYNC_PROCESSING="1"
$env:DISABLE_RAG_STARTUP="1"
$env:CANVAS_APP_ID="theme11-local"
$env:APP_SESSION_SECRET="dev-session-secret-change-me"
$env:PROGRESS_HMAC_SECRET="dev-progress-secret-change-me"
python main.py
```

Open `http://localhost:5000/judgments`.

For the local UI, set identity in browser storage:

```javascript
localStorage.setItem("canvas_user_id", "demo-user")
```

If API-key auth is enabled in your environment, also set:

```javascript
localStorage.setItem("canvas_api_key", "your-api-key")
```

## Demo Modes

`JUDGMENT_SYNC_PROCESSING=1` is the recommended hackathon demo mode. It processes uploads directly in FastAPI, so Redis/Celery is not needed.

`DISABLE_RAG_STARTUP=1` is also recommended for the Theme 11 demo. It skips eager general-RAG model loading so the judgment workflow starts cleanly on offline or restricted networks.

Celery mode is still supported:

```powershell
redis-server
celery -A celery_worker.celery_app worker --loglevel=info
uvicorn main:app --host 0.0.0.0 --port 5000
```

Use `POST /judgments/demo-seed?user_id=demo-user` or the UI button to create a deterministic synthetic record.

## Theme 11 Endpoints

- `POST /judgments/upload?sync=true`
- `POST /judgments/from-ccms?sync=true`
- `POST /judgments/demo-seed?user_id=demo-user`
- `GET /judgments/records/{user_id}`
- `GET /judgments/dashboard/{user_id}`
- `GET /judgments/dashboard/{user_id}/export?format=csv`
- `GET /judgments/{record_id}?user_id={user_id}`
- `POST /judgments/{record_id}/review?user_id={user_id}`
- `POST /judgments/{record_id}/ask`
- `GET /judgments/{record_id}/audit?user_id={user_id}`
- `GET /judgments/{record_id}/highlighted-pdf?user_id={user_id}`

## OCR Note

Digital PDFs are parsed directly when the embedded text layer is reliable. Low-text, scanned, or corrupted text-layer PDFs are profiled and routed through local MiniCPM vision OCR via Ollama (`openbmb/minicpm-o2.6:latest`) when OCR is needed; corrupted embedded text is discarded instead of being trusted.

## Evaluation Harness

Run repeatable accuracy benchmarks across judgment PDFs using `eval_harness.py`.

### Batch evaluation (no ground truth)

Evaluate every PDF in a directory and save JSON outputs:

```powershell
python eval_harness.py --pdf-dir user_data/evaluation_inputs --output-dir user_data/evaluation_outputs --llm-enabled false
```

### Benchmark with expected outputs

Create a manifest JSON that points to PDFs and their expected field/action values:

```powershell
python eval_harness.py --manifest user_data/evaluation_inputs/sample_manifest.json --output-dir user_data/evaluation_outputs --llm-enabled false
```

Manifest format (see `user_data/evaluation_inputs/sample_manifest.json`):

```json
[
    {
        "pdf_path": "user_data/evaluation_inputs/demo_writ_petition.pdf",
        "metadata_json": "{\"dataset\":\"theme11-sandbox\"}",
        "expected": {
            "case_number": "Writ Petition No. 1234 of 2025",
            "court": "High Court Of Karnataka At Bengaluru",
            "judgment_date": "2026-03-15",
            "departments": ["BBMP", "Urban Development Department"],
            "disposition": "Allowed",
            "action_items": [
                {"title": "Remove the encroachment", "owner": "BBMP", "category": "compliance", "priority": "high"}
            ],
            "directions": [
                {"text": "The BBMP is directed to remove the encroachment within four weeks."}
            ],
            "metrics": {
                "evidence_coverage_percent": 100,
                "ambiguous_count": 0,
                "duplicate_count": 0,
                "ocr_used": false
            }
        }
    }
]
```

The harness scores:
- **Fields**: fuzzy text match, date normalisation, set overlap for lists
- **Action items**: title match (fuzzy), then owner/category/priority
- **Directions**: text match against extracted direction values
- **Quality metrics**: exact or ±5% tolerance for coverage

### Local evaluation without HTTP server

```powershell
python eval_harness.py --local --manifest user_data/evaluation_inputs/sample_manifest.json --output-dir user_data/evaluation_outputs --llm-enabled false
```

### Output

Each run produces:
- `{stem}_response.json` — raw `judgment-evaluation-v1` response
- `{stem}_agent_brief.md` — compact Markdown packet for another AI agent to summarize or compare by hand
- `{stem}_meta.json` — per-file score breakdown
- `summary.json` — aggregate accuracy across all files

### Deterministic regression baseline

Use `--llm-enabled false` for deterministic extraction baselines. Re-run with `--llm-enabled true` to compare LLM enrichment impact.

## Verification

```powershell
python -m pytest tests/test_judgment_core.py tests/test_judgment_workflow.py tests/test_eval_harness.py -q
python -m py_compile main.py celery_worker.py judgment_workflow/api.py judgment_workflow/pipeline.py judgment_workflow/repository.py rag/judgment/extractor.py rag/judgment/action_plan.py eval_harness.py
node --check frontend/judgments.js
```

## Prototype Constraints

- CCMS integration has a real HTTP adapter shape plus a mock fallback; production credentials and government sandbox contracts are still required.
- LLM enrichment is guardrailed and optional; deterministic extraction remains the baseline.
- Reviewer approval is evidence-gated, but final legal responsibility remains with the human reviewer.
