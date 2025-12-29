# AI Engineering

This repository contains example code from explorations in AI engineering. Run `uv sync` to install necessary dependencies.

## Streaming Data Pipeline

[streaming-data-pipeline.py](streaming-data-pipeline.py)

Implements a simulated streaming data pipeline using Python's asyncio for real-time event processing. It demonstrates how to efficiently track statistics—such as unique users, user frequencies, value distributions, rolling mean/variance, and latency percentiles—on high-throughput data streams. Probabilistic data structures like HyperLogLog and Count-Min Sketch are used for scalable, memory-efficient estimation of unique counts and item frequencies, which are critical in large-scale systems where exact computation is costly. 

This approach is useful for studying real-world scenarios like monitoring user activity, detecting anomalies, or analyzing metrics in web services, IoT, or financial platforms. In production, the pipeline could be extended to handle distributed sources, integrate with message brokers, and persist results for further analytics.

## Streaming API

[streaming-backend.py](streaming-backend.py)

Implements a streaming API using FastAPI and asyncio. It provides endpoints for creating jobs, retrieving job status, and cancelling jobs. It supports real-time updates through Server-Sent Events (SSE), allowing clients to see results as they become available. 

This is motivated by LLM applications where responses can be streamed incrementally, enhancing user experience by reducing perceived latency. In production, the app could be extended with authentication, persistent storage, and integration with background workers or backend services for doing actual processing. Rather than sending the entire result with each update, it would likely send small deltas (chunks/batches). With replay buffers late subscribers could catch up.

Run with: `uvicorn async-backend:app --reload`

[streaming-frontend.html](streaming-frontend.html)

Open in a web browser to demo the API.