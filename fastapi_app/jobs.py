"""In-process background jobs with progress reporting (thread pool)."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from . import settings

log = logging.getLogger("synth_re.jobs")

ProgressFn = Callable[[float, str], None]


class JobManager:
    def __init__(self, workers: int = 2, keep: int = 50):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._keep = keep

    def submit(self, fn: Callable[[ProgressFn], Any], scenario_id: Optional[str] = None) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "scenario_id": scenario_id, "status": "queued", "progress": 0.0,
               "message": "Queued", "result_id": None, "error": None, "created": time.time()}
        with self._lock:
            self._jobs[job_id] = job
            while len(self._jobs) > self._keep:
                self._jobs.popitem(last=False)
        log.info("job %s submitted (scenario %s)", job_id, scenario_id)
        self._pool.submit(self._run, job_id, fn)
        return self.get(job_id)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def _run(self, job_id: str, fn: Callable[[ProgressFn], Any]) -> None:
        log.info("job %s started", job_id)
        self._update(job_id, status="running", message="Starting…")

        def progress(fraction: float, message: str = "") -> None:
            self._update(job_id, progress=max(0.0, min(1.0, float(fraction))), message=message)

        try:
            result_id = fn(progress)
            log.info("job %s done -> result %s", job_id, result_id)
            self._update(job_id, status="done", progress=1.0, message="Done", result_id=result_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, logged with traceback
            log.exception("job %s failed", job_id)
            self._update(job_id, status="error", error=str(exc) or exc.__class__.__name__, message="Failed")


manager = JobManager(workers=settings.JOB_WORKERS)
