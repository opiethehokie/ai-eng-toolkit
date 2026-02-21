# AI Engineering

Small, focused examples from my AI engineering explorations. This repo is meant to be a portfolio of practical patterns, not a framework.

## Setup

- Python `>=3.13`
- Install deps: `uv sync`
- For LangChain demos: set `OPENAI_API_KEY` (via `.env` or env vars)

## Streaming API (LLM-style SSE)

- Backend: `streaming-backend.py`  
  Single `POST /stream` endpoint that streams tokens via Server-Sent Events, supports `Last-Event-ID` resume, and emits `ping/token/done` events.
- Frontend: `streaming-frontend.html`  
  Minimal fetch + `ReadableStream` client with stop/resume controls.

```
Client UI            FastAPI /stream
   |  POST (text)         |
   |--------------------->|
   |  SSE: ping           |
   |<---------------------|
   |  SSE: token ...      |
   |<---------------------|
   |  SSE: token ...      |
   |<---------------------|
   |  SSE: done           |
   |<---------------------|
```

Run the backend: `uvicorn streaming-backend:app --reload`  
Open the HTML file directly in a browser to demo.

## Realtime Audio (WebRTC + TypeScript)

Requires Node.js 20+

- Backend: `realtime-audio/src/server.ts`  
  Express server with `GET /token` to mint ephemeral Realtime sessions and static hosting for the demo page.
- Frontend: `realtime-audio/public/index.html`  
  Single-page push-to-talk voice chat client using WebRTC data channel events for commit/response and interruption (`response.cancel` + `output_audio_buffer.clear`).

Run the backend:

```bash
cd realtime-audio
npm install
npm run dev
```

Access the frontend at http://localhost:3000/index.html

## Streaming Data Pipeline

- `streaming-data-pipeline.py`  
  Simulated real-time pipeline using `asyncio` that tracks unique users, counts, value distributions, rolling stats, and latency percentiles. Uses probabilistic sketches (HyperLogLog, Count-Min Sketch) for scale-friendly estimates.

## Multi-Agent Patterns (LangChain + LangGraph)

- `langchain-multi-agent-patterns/router.py`  
  Router pattern: classify a query and route to specialized agents, then synthesize results.
- `langchain-multi-agent-patterns/supervisor.py`  
  Supervisor pattern: a central coordinator delegates to expert agents.
- `langchain-multi-agent-patterns/state-machine.py`  
  State machine pattern: a single agent changes behavior across workflow steps.
- `langchain-multi-agent-patterns/skills.py`  
  Progressive disclosure of skills: load only needed instructions on demand.
