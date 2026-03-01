# AI Engineering Toolkit

Small demo projects that capture practical AI engineering patterns. This repo is intentionally not a framework. It is a set of runnable references for recurring problems: routing, streaming, durability, realtime UX, and context management.

## Setup

### Prerequisites

- Python `>=3.13`
- `uv` for Python dependency management
- Optional: Node.js `>=20` (for `realtime-audio/`)
- Optional: Temporal CLI (for the Temporal demo)
- Optional: `OPENAI_API_KEY` in `.env` (for LLM-backed demos)

### Install

```bash
uv sync
```

## Repo map (what to open first)

- `litellm-gateway.py`: Marimo notebook for retry/fallback/escalation gateway behavior
- `streaming-backend.py`: FastAPI SSE token-stream backend with resume support
- `streaming-frontend.html`: zero-build browser client for SSE streaming demo
- `streaming-data-pipeline.py`: asyncio stream processing + live terminal dashboard
- `temporal-agent.py`: durable incident triage workflow with human approval signal
- `langchain-multi-agent-patterns/`: four multi-agent architecture patterns
- `realtime-audio/`: push-to-talk WebRTC demo with OpenAI Realtime API

---

## 1) LiteLLM Gateway Demo (Retries, Fallbacks, Quality Escalation)

### Files

- `litellm-gateway.py` (Marimo app)

### What it demonstrates

- Cost-first primary model routing
- Retry behavior for transient failures
- Fallback to higher-quality model when retries fail
- Optional quality-based escalation even after a successful primary call
- Deterministic fault injection so failure paths are demoable on demand

### Why it’s structured this way

- Real gateways fail in non-happy-path ways; the notebook makes those paths explicit.
- `Router` is used directly to keep retry/fallback behavior close to what production gateways do.
- Quality escalation is separated from retry/fallback to show that “success” and “sufficient quality” are different decisions.

### Run

```bash
marimo run litellm-gateway.py
```

---

## 2) Streaming API Demo (SSE, LLM-style token delivery)

### Files

- `streaming-backend.py`
- `streaming-frontend.html`

### What it demonstrates

- `POST /stream` emits SSE frames: `ping`, `token`, `done`
- `Last-Event-ID` resume handling for interrupted streams
- Browser client using `fetch` + `ReadableStream` (not `EventSource`, because request body is needed)

### Why it’s structured this way

- Mirrors how modern LLM streaming endpoints behave, but in a tiny inspectable form.
- Resume support is included because flaky networks are normal in real clients.
- Initial `ping` reduces perceived latency and confirms stream setup quickly.

### Run

```bash
uvicorn streaming-backend:app --reload
```

Then open `streaming-frontend.html` directly in a browser and point at `http://127.0.0.1:8000/stream` (already set in the file).

---

## 3) Realtime Audio Demo (WebRTC Push-to-Talk)

### Files

- `realtime-audio/src/server.ts`: Express server + token minting proxy
- `realtime-audio/public/index.html`: simple UI
- `realtime-audio/public/app.js`: WebRTC + realtime event protocol handling

### What it demonstrates

- Server-side minting of ephemeral realtime session keys via `GET /token`
- Browser microphone capture + WebRTC negotiation
- Push-to-talk turns, interruption handling (`response.cancel` + `output_audio_buffer.clear`)
- Transcript/debug event display in UI

### Why it’s structured this way

- Keeps permanent API keys off the browser.
- Uses a minimal static server to reduce moving parts while preserving realistic auth flow.
- Push-to-talk control makes turn boundaries explicit, which is useful for demos and debugging.

### Run

```bash
cd realtime-audio
npm install
npm run dev
```

Open [http://localhost:3000/index.html](http://localhost:3000/index.html)

---

## 4) Streaming Data Pipeline Demo

### Files

- `streaming-data-pipeline.py`

### What it demonstrates

- Async pub/sub simulation with bounded queue backpressure
- Batch processing by size or timeout
- Rolling stats and latency percentiles
- Approximate-at-scale counting via HyperLogLog and Count-Min Sketch
- Live terminal dashboard + dynamic p99 anomaly threshold

### Why it’s structured this way

- Emphasizes operations-style observability (latency, queue depth, anomalies), not just correctness.
- Uses probabilistic sketches to show scale-minded patterns without large infrastructure.

### Run

```bash
python streaming-data-pipeline.py
```

---

## 5) Multi-Agent Patterns (LangChain + LangGraph)

### Files

- `langchain-multi-agent-patterns/router.py`
- `langchain-multi-agent-patterns/supervisor.py`
- `langchain-multi-agent-patterns/state-machine.py`
- `langchain-multi-agent-patterns/skills.py`

### What each pattern is for

- `router.py`: classify a query, fan out to domain agents, synthesize answer
- `supervisor.py`: central coordinator delegates to specialist sub-agents/tools
- `state-machine.py`: one agent whose prompt/tools change by workflow step
- `skills.py`: progressive disclosure; load detailed skill context only when needed

### Why these examples exist

- They capture four commonly-confused architectures side-by-side.
- Each file is intentionally standalone so tradeoffs are visible without framework noise.

### Run (any example)

```bash
python langchain-multi-agent-patterns/router.py
python langchain-multi-agent-patterns/supervisor.py
python langchain-multi-agent-patterns/state-machine.py
python langchain-multi-agent-patterns/skills.py
```

Requires `OPENAI_API_KEY`.

---

## 6) Durable Agent Workflow Demo (Temporal + Human Approval)

### Files

- `temporal-agent.py`

### What it demonstrates

- Durable workflow orchestration around incident triage
- Activity retries with explicit retry policy
- Human-in-the-loop approval via workflow signal
- Timeout-based escalation when no approval arrives
- Workflow history replay validation for determinism checks
- Swappable triage mode: deterministic mock vs LLM-backed agent

### Why it’s structured this way

- Durable execution matters most where human latency and system retries collide.
- The workflow separates “decision generation” from “remediation execution” to make governance points explicit.
- Replay is included because durable systems fail silently if nondeterminism sneaks in.

### Run

Install Temporal CLI: [https://temporal.io/setup/install-temporal-cli](https://temporal.io/setup/install-temporal-cli)

```bash
# terminal 1
temporal server start-dev --ip 127.0.0.1 --port 7233

# terminal 2
python temporal-agent.py worker

# terminal 3
python temporal-agent.py start --wait
```

Approve from another terminal:

```bash
python temporal-agent.py approve --workflow-id <workflow-id> --reviewer you --note "approved for demo"
```

Replay:

```bash
python temporal-agent.py replay --workflow-id <workflow-id>
```

Use LLM triage mode instead of mock:

```bash
export TRIAGE_AGENT_MODE=openai
