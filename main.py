from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from judgment_workflow.api import judgment_router
from storage import close_storage, init_storage


load_dotenv()

os.environ.setdefault("STORAGE_BACKEND", "sqlite")
os.environ.setdefault("SQLITE_DB_PATH", "user_data/local_demo/artha.db")
os.environ.setdefault("JUDGMENT_DATA_ROOT", "user_data/judgments")
os.environ.setdefault("JUDGMENT_SYNC_PROCESSING", "1")
os.environ.setdefault("JUDGMENT_DEMO_AUTH_BYPASS", "1")
os.environ.setdefault("CANVAS_APP_ID", "theme11-local")

APP_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = APP_ROOT / "frontend"

app = FastAPI(
    title="Artha Judge",
    description="AI for Bharat Theme 11 prototype: judgment PDFs to verified action plans.",
    version="0.1.0",
)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_ROOT)), name="frontend")
app.include_router(judgment_router)


@app.on_event("startup")
async def startup() -> None:
    await init_storage()


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_storage()


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/judgments")


@app.get("/judgments", include_in_schema=False)
async def judgments_page():
    return FileResponse(FRONTEND_ROOT / "judgments.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=False)
