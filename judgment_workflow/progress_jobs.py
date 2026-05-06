from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


STAGE_DEFINITIONS = [
    ("upload", "Upload received", 5.0),
    ("judgment_processing", "Reading judgment", 15.0),
    ("ocr_detection", "Checking OCR", 35.0),
    ("judgment_extraction", "Extracting legal facts", 55.0),
    ("highlight_generation", "Building source proof", 75.0),
    ("record_storage", "Saving review package", 90.0),
    ("complete", "Ready for human review", 100.0),
]


@dataclass
class ProgressJob:
    job_id: str
    record_id: str
    state: str = "queued"
    stage: str = "upload"
    message: str = "Upload received. Preparing the AI workflow..."
    pct: float = 5.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ProgressJobStore:
    def __init__(self, *, max_jobs: int = 50, ttl_seconds: int = 3600) -> None:
        self.max_jobs = max_jobs
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, ProgressJob] = {}
        self._lock = threading.RLock()

    def create(self, *, record_id: str) -> ProgressJob:
        with self._lock:
            self._prune_locked()
            job = ProgressJob(job_id=uuid.uuid4().hex, record_id=record_id)
            self._jobs[job.job_id] = job
            return job

    def get(self, job_id: str) -> ProgressJob | None:
        with self._lock:
            self._prune_locked()
            return self._jobs.get(job_id)

    def update(self, job_id: str, **updates: Any) -> ProgressJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            return job

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [job_id for job_id, job in self._jobs.items() if now - job.updated_at > self.ttl_seconds]
        for job_id in expired:
            self._jobs.pop(job_id, None)
        if len(self._jobs) <= self.max_jobs:
            return
        by_age = sorted(self._jobs.values(), key=lambda job: job.updated_at)
        for job in by_age[: len(self._jobs) - self.max_jobs]:
            self._jobs.pop(job.job_id, None)


progress_jobs = ProgressJobStore()


def serialize_job(job: ProgressJob) -> dict[str, Any]:
    active_index = _stage_index(job.stage)
    stages = []
    for index, (key, label, _) in enumerate(STAGE_DEFINITIONS):
        if job.state == "failure":
            state = "error" if index == active_index else ("complete" if index < active_index else "pending")
        elif job.state == "success":
            state = "complete"
        elif index < active_index:
            state = "complete"
        elif index == active_index:
            state = "active"
        else:
            state = "pending"
        stages.append({"key": key, "label": label, "state": state})
    return {
        "job_id": job.job_id,
        "record_id": job.record_id,
        "state": job.state,
        "stage": job.stage,
        "message": job.message,
        "pct": job.pct,
        "stages": stages,
        "result": job.result,
        "error": job.error,
    }


def start_progress_task(
    *,
    job: ProgressJob,
    process: Callable[[Callable[..., None]], Awaitable[dict[str, Any]]],
) -> None:
    def runner() -> None:
        progress_jobs.update(job.job_id, state="running")

        def progress_callback(*, stage: str, message: str, pct: float) -> None:
            progress_jobs.update(
                job.job_id,
                state="running",
                stage=stage,
                message=message,
                pct=float(pct),
            )

        try:
            result = asyncio.run(process(progress_callback))
        except Exception as exc:
            progress_jobs.update(
                job.job_id,
                state="failure",
                stage="failed",
                message="Judgment processing failed.",
                error=str(exc),
            )
            return

        progress_jobs.update(
            job.job_id,
            state="success",
            stage="complete",
            message="Judgment workflow complete.",
            pct=100.0,
            result={"record_id": result.get("record_id", job.record_id)},
        )

    thread = threading.Thread(target=runner, name=f"judgment-progress-{job.job_id[:8]}", daemon=True)
    thread.start()


def _stage_index(stage: str) -> int:
    keys = [key for key, _, _ in STAGE_DEFINITIONS]
    if stage == "failed":
        return max(0, len(keys) - 1)
    try:
        return keys.index(stage)
    except ValueError:
        return 0
