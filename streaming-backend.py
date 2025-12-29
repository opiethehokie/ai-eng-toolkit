import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",  # needed if you open index.html via file://
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models (typed metadata) ----------


class JobStatus(str, Enum):
    pending = "pending"
    active = "active"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"


class JobRequest(BaseModel):
    text: str = Field(..., description="Input to process")
    steps: int = Field(5, ge=1, le=50, description="How many progress steps to emit")
    delay_ms: int = Field(250, ge=0, le=5000, description="Artificial delay per step")


class JobResult(BaseModel):
    output: str
    took_ms: int


class JobPublic(BaseModel):
    id: str
    status: JobStatus
    created_at_ms: int
    updated_at_ms: int
    request: JobRequest
    result: Optional[JobResult] = None
    error: Optional[str] = None


class StreamEvent(BaseModel):
    job_id: str
    seq: int
    kind: str  # "progress" | "completed" | "cancelled" | "failed" | "ping"
    ts_ms: int
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[JobResult] = None


# ---------- In-memory job store ----------


@dataclass
class Job:
    id: str
    created_at_ms: int
    updated_at_ms: int
    status: JobStatus
    request: JobRequest
    result: Optional[JobResult] = None
    error: Optional[str] = None

    events: asyncio.Queue[StreamEvent] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task[None]] = None


JOBS: dict[str, Job] = {}


def now_ms() -> int:
    return int(time.time() * 1000)


def to_public(job: Job) -> JobPublic:
    return JobPublic(
        id=job.id,
        status=job.status,
        created_at_ms=job.created_at_ms,
        updated_at_ms=job.updated_at_ms,
        request=job.request,
        result=job.result,
        error=job.error,
    )


# ---------- “Work” implementation ----------


async def run_job(job: Job) -> None:
    job.status = JobStatus.active
    job.updated_at_ms = now_ms()

    start = now_ms()
    try:
        steps = job.request.steps
        for i in range(1, steps + 1):
            if job.cancel_event.is_set():
                job.status = JobStatus.cancelled
                job.updated_at_ms = now_ms()
                await job.events.put(StreamEvent(job_id=job.id, seq=i, kind="cancelled", ts_ms=job.updated_at_ms, message="Cancelled by client."))
                return

            await asyncio.sleep(job.request.delay_ms / 1000.0)

            partial_out = job.request.text.upper()[: math.floor(len(job.request.text) * i / (steps + 1))]  # simulated partial result client can show
            await job.events.put(
                StreamEvent(
                    job_id=job.id,
                    seq=i,
                    kind="progress",
                    ts_ms=now_ms(),
                    progress=i / steps,
                    result=JobResult(output=partial_out, took_ms=now_ms() - start),
                    message=f"Step {i}/{steps}",
                )
            )

        # simulated final result
        out = job.request.text.upper()
        took = now_ms() - start
        job.result = JobResult(output=out, took_ms=took)
        job.status = JobStatus.completed
        job.updated_at_ms = now_ms()

        await job.events.put(
            StreamEvent(
                job_id=job.id,
                seq=steps + 1,
                kind="completed",
                ts_ms=job.updated_at_ms,
                result=job.result,
                message="Done.",
            )
        )

    except Exception as e:
        job.status = JobStatus.failed
        job.error = str(e)
        job.updated_at_ms = now_ms()
        await job.events.put(StreamEvent(job_id=job.id, seq=-1, kind="failed", ts_ms=job.updated_at_ms, message=job.error))


def sse_encode(event: StreamEvent) -> str:
    # minimal SSE framing; we send JSON as the `data:` line
    data = event.model_dump_json()
    return f"event: {event.kind}\ndata: {data}\n\n"


# ---------- Endpoints ----------


@app.post("/sync")
def sync_endpoint(body: JobRequest) -> dict[str, Any]:
    # same “thing” as async: produce uppercased output + timing + metadata
    start = now_ms()
    time.sleep(body.steps * body.delay_ms / 1000.0)
    out = body.text.upper()
    took = now_ms() - start
    return {
        "output": out,
        "meta": {
            "took_ms": took,
            "echo_steps": body.steps,
            "echo_delay_ms": body.delay_ms,
        },
    }


@app.post("/jobs", response_model=JobPublic)
async def create_job(body: JobRequest) -> JobPublic:
    job_id = uuid.uuid4().hex
    t = now_ms()
    job = Job(
        id=job_id,
        created_at_ms=t,
        updated_at_ms=t,
        status=JobStatus.pending,
        request=body,
    )
    JOBS[job_id] = job

    # schedule background work
    job.task = asyncio.create_task(run_job(job))
    return to_public(job)


@app.get("/jobs/{job_id}", response_model=JobPublic)
def get_job(job_id: str) -> JobPublic:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return to_public(job)


@app.post("/jobs/{job_id}/cancel", response_model=JobPublic)
async def cancel_job(job_id: str) -> JobPublic:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")

    job.cancel_event.set()
    job.updated_at_ms = now_ms()

    # If it’s waiting somewhere, let the stream see a cancellation quickly.
    await job.events.put(StreamEvent(job_id=job.id, seq=-1, kind="cancelled", ts_ms=job.updated_at_ms, message="Cancel requested."))
    return to_public(job)


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")

    async def gen():
        # Optional initial snapshot event
        yield sse_encode(
            StreamEvent(
                job_id=job.id,
                seq=0,
                kind="progress",
                ts_ms=now_ms(),
                progress=0.0,
                message=f"Subscribed. status={job.status}",
            )
        )

        # Keep streaming until terminal state and queue drained or client disconnects
        while True:
            if await request.is_disconnected():
                # client closed tab / connection dropped
                job.cancel_event.set()
                break

            try:
                evt = await asyncio.wait_for(job.events.get(), timeout=10.0)
                yield sse_encode(evt)

                if evt.kind in ("completed", "failed", "cancelled"):
                    break

            except asyncio.TimeoutError:
                # keepalive so proxies don’t buffer forever
                yield sse_encode(StreamEvent(job_id=job.id, seq=-999, kind="ping", ts_ms=now_ms(), message="keepalive"))

    return StreamingResponse(gen(), media_type="text/event-stream")
