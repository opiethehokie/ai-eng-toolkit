import asyncio
import time
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
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


class StreamRequest(BaseModel):
    text: str = Field(..., description="Input to stream back as tokens")
    delay_ms: int = Field(60, ge=0, le=2000, description="Artificial delay per token")


def now_ms() -> int:
    return int(time.time() * 1000)


def sse(event: str, data: str, event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {data}")
    return "\n".join(lines) + "\n\n"


async def token_stream(text: str, delay_ms: int, last_event_id: Optional[int]) -> AsyncIterator[str]:
    start_ms = now_ms()
    tokens = text.split(" ")
    if len(tokens) == 1 and tokens[0] == "":
        done = {"elapsed_ms": now_ms() - start_ms, "tokens": 0}
        yield sse("done", json_dumps(done), event_id=1)
        return
    token_count = len(tokens)
    if last_event_id is not None and last_event_id >= token_count + 1:
        done = {"elapsed_ms": now_ms() - start_ms, "tokens": token_count}
        yield sse("done", json_dumps(done), event_id=token_count + 1)
        return

    start_index = 1
    if last_event_id is not None:
        start_index = max(1, last_event_id + 1)

    for i in range(start_index, token_count + 1):
        tok = tokens[i - 1]
        await asyncio.sleep(delay_ms / 1000.0)
        payload = {
            "token": tok + (" " if i < len(tokens) else ""),
            "index": i,
            "elapsed_ms": now_ms() - start_ms,
        }
        yield sse("token", json_dumps(payload), event_id=i)
    done = {"elapsed_ms": now_ms() - start_ms, "tokens": len(tokens)}
    yield sse("done", json_dumps(done), event_id=len(tokens) + 1)


def json_dumps(obj: object) -> str:
    # tiny local wrapper so we don't have to import a full JSON encoder class
    import json

    return json.dumps(obj, separators=(",", ":"))


@app.post("/stream")
async def stream_endpoint(body: StreamRequest, request: Request):
    last_event_id = request.headers.get("last-event-id")
    last_id = None
    if last_event_id is not None:
        try:
            last_id = int(last_event_id)
        except ValueError:
            last_id = None

    async def gen() -> AsyncIterator[str]:
        # initial ping to reduce time-to-first-byte
        yield sse("ping", json_dumps({"ts_ms": now_ms()}), event_id=0)

        async for chunk in token_stream(body.text, body.delay_ms, last_id):
            if await request.is_disconnected():
                break
            yield chunk

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
