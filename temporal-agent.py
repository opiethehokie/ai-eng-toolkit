"""Temporal incident-triage demo.

Quick concept map:
- Workflow: durable orchestration logic that Temporal replays from event history.
- Activity: side-effecting step (API call, DB write, LLM call) invoked by a workflow.
- Signal: external async input sent to a running workflow (used here for human approval).
- Replay: re-run workflow code against stored history to verify determinism.

Why Temporal in this project:
- This demo includes retries, long waits, and human-in-the-loop approval.
- Temporal persists workflow state/history so a worker crash does not lose progress.
- We avoid writing custom "resume from DB", retry schedulers, and timeout plumbing.
- Net effect: we focus on business logic while Temporal handles durability mechanics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Replayer, Worker

load_dotenv()

TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
TASK_QUEUE = os.getenv("TEMPORAL_TASK_QUEUE", "incident-triage")
AGENT_MODE = os.getenv("TRIAGE_AGENT_MODE", "mock").lower()
AGENT_MODEL = os.getenv("TRIAGE_AGENT_MODEL", "gpt-4.1-mini")


def _mock_triage(incident: dict[str, Any]) -> dict[str, Any]:
    summary = str(incident.get("summary", "")).lower()
    ratio = float(incident["metric_value"]) / max(float(incident["threshold"]), 0.1)
    if "outage" in summary or ratio >= 3:
        severity = "critical"
    elif any(token in summary for token in ("error", "latency", "degraded")) or ratio >= 1.8:
        severity = "high"
    elif ratio >= 1.2:
        severity = "medium"
    else:
        severity = "low"

    return {
        "severity": severity,
        "confidence": 0.75 if severity in {"high", "critical"} else 0.9,
        "summary": f"{incident['service']} is {severity}.",
        "remediation_steps": [
            "Validate signal with logs and metrics.",
            "Check recent deploy/config changes.",
            "Mitigate blast radius.",
        ],
        "needs_human_approval": severity in {"high", "critical"},
    }


async def _agent_triage(incident: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    # Demo switch: keep workflows runnable without external LLM dependencies.
    if AGENT_MODE == "mock" or not os.getenv("OPENAI_API_KEY"):
        return _mock_triage(incident)

    try:
        from agents import Agent, Runner
    except ImportError:
        return _mock_triage(incident)

    prompt = {
        "incident": incident,
        "context": context,
        "output_contract": {
            "severity": "low|medium|high|critical",
            "confidence": "0..1",
            "summary": "short paragraph",
            "remediation_steps": ["step"],
            "needs_human_approval": True,
        },
    }
    agent = Agent(name="incident-triage", model=AGENT_MODEL, instructions="Return strict JSON only.")
    result = await Runner.run(agent, input=json.dumps(prompt))
    output = result.final_output

    if isinstance(output, dict):
        payload = output
    else:
        cleaned = str(output).replace("```json", "").replace("```", "").strip()
        payload = json.loads(cleaned)

    return {
        "severity": str(payload.get("severity", "medium")),
        "confidence": float(payload.get("confidence", 0.5)),
        "summary": str(payload.get("summary", "No summary provided.")),
        "remediation_steps": [str(s) for s in payload.get("remediation_steps", [])],
        "needs_human_approval": bool(payload.get("needs_human_approval", True)),
    }


# Activity = effectful unit of work. Temporal can retry it independently (should be idempotent).
@activity.defn
async def fetch_context(incident: dict[str, Any]) -> dict[str, Any]:
    activity.logger.info("Fetching context for %s", incident["incident_id"])
    return {
        "recent_changes": ["deploy: payments-api v2026.02.22.1"],
        "logs_excerpt": "Timeout-heavy errors around upstream boundary",
    }


# Activity = effectful unit of work. Kept separate so failures/retries are isolated.
@activity.defn
async def triage(incident: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return await _agent_triage(incident, context)


# Activity = side-effect boundary where remediation would call real systems.
@activity.defn
async def remediate(incident: dict[str, Any], decision: dict[str, Any]) -> str:
    first_step = (decision.get("remediation_steps") or ["No step"])[0]
    return f"Executed remediation for {incident['service']}: {first_step}"


# Workflow = durable control plane. Should orchestrate; avoid doing side effects directly here.
@workflow.defn
class IncidentTriageWorkflow:
    def __init__(self) -> None:
        self.approval: dict[str, str] | None = None

    @workflow.run
    async def run(self, incident: dict[str, Any], approval_timeout_seconds: int = 300) -> dict[str, Any]:
        # Workflow code is your durable state machine: Temporal replays this logic from history.
        # Activities isolate side effects; RetryPolicy demonstrates durable retry semantics.
        context = await workflow.execute_activity(
            fetch_context,
            args=[incident],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        decision = await workflow.execute_activity(
            triage,
            args=[incident, context],
            start_to_close_timeout=timedelta(seconds=90),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Workflows can "wait forever" safely; Temporal persists state between waits.
        if decision.get("needs_human_approval", False):
            deadline = workflow.now() + timedelta(seconds=approval_timeout_seconds)
            # Human-in-the-loop gate: workflow parks until signal arrives or timeout expires.
            # This wait is resilient to restarts because time/progress are recorded in workflow history.
            while self.approval is None and workflow.now() < deadline:
                await workflow.sleep(5)

            if self.approval is None:
                remediation_result = "Escalated: no approval before timeout"
                final_status = "timed_out"
            else:
                remediation_result = await workflow.execute_activity(
                    remediate,
                    args=[incident, decision],
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                final_status = "completed"
        else:
            remediation_result = await workflow.execute_activity(
                remediate,
                args=[incident, decision],
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            final_status = "completed"

        return {
            "workflow_id": workflow.info().workflow_id,
            "run_id": workflow.info().run_id,
            "incident_id": incident["incident_id"],
            "decision": decision,
            "approval": self.approval,
            "remediation_result": remediation_result,
            "final_status": final_status,
        }

    @workflow.signal
    def approve(self, reviewer: str, note: str = "approved") -> None:
        # Signal handler: asynchronous external input that mutates workflow state.
        self.approval = {"reviewer": reviewer, "note": note}


async def cmd_worker(args: argparse.Namespace) -> None:
    client = await Client.connect(args.address)
    # Worker polls task queue and executes both workflow tasks and activity tasks.
    # If a worker dies mid-run, a new worker can continue from persisted workflow history.
    worker = Worker(
        client,
        task_queue=args.task_queue,
        workflows=[IncidentTriageWorkflow],
        activities=[fetch_context, triage, remediate],
    )
    print(f"Worker listening on {args.address} queue={args.task_queue}")
    await worker.run()


async def cmd_start(args: argparse.Namespace) -> None:
    client = await Client.connect(args.address)
    workflow_id = args.workflow_id or f"incident-triage-{uuid.uuid4().hex[:8]}"
    incident = {
        "incident_id": args.incident_id,
        "service": args.service,
        "summary": args.summary,
        "metric_value": args.metric_value,
        "threshold": args.threshold,
    }
    handle = await client.start_workflow(
        IncidentTriageWorkflow.run,
        args=[incident, args.approval_timeout_seconds],
        id=workflow_id,
        task_queue=args.task_queue,
    )
    print(f"started workflow_id={handle.id}")
    if args.wait:
        print(json.dumps(await handle.result(), indent=2))


async def cmd_approve(args: argparse.Namespace) -> None:
    client = await Client.connect(args.address)
    handle = client.get_workflow_handle(args.workflow_id)
    # Signal from CLI to running workflow instance.
    await handle.signal(IncidentTriageWorkflow.approve, args=[args.reviewer, args.note])
    print("approval signal sent")


async def cmd_replay(args: argparse.Namespace) -> None:
    client = await Client.connect(args.address)
    history = await client.get_workflow_handle(args.workflow_id).fetch_history()
    # Replay is the determinism check: workflow code must reproduce decisions from history.
    await Replayer(workflows=[IncidentTriageWorkflow]).replay_workflow(history)
    print(f"replay passed for workflow_id={args.workflow_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal durable incident triage demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    worker_p = sub.add_parser("worker")
    worker_p.add_argument("--address", default=TEMPORAL_ADDRESS)
    worker_p.add_argument("--task-queue", default=TASK_QUEUE)

    start_p = sub.add_parser("start")
    start_p.add_argument("--address", default=TEMPORAL_ADDRESS)
    start_p.add_argument("--task-queue", default=TASK_QUEUE)
    start_p.add_argument("--workflow-id", default=None)
    start_p.add_argument("--incident-id", default=f"inc-{uuid.uuid4().hex[:8]}")
    start_p.add_argument("--service", default="payments-api")
    start_p.add_argument("--summary", default="Error rate spike after deploy")
    start_p.add_argument("--metric-value", type=float, default=8.4)
    start_p.add_argument("--threshold", type=float, default=3.0)
    start_p.add_argument("--approval-timeout-seconds", type=int, default=300)
    start_p.add_argument("--wait", action="store_true")

    approve_p = sub.add_parser("approve")
    approve_p.add_argument("--address", default=TEMPORAL_ADDRESS)
    approve_p.add_argument("--workflow-id", required=True)
    approve_p.add_argument("--reviewer", required=True)
    approve_p.add_argument("--note", default="approved")

    replay_p = sub.add_parser("replay")
    replay_p.add_argument("--address", default=TEMPORAL_ADDRESS)
    replay_p.add_argument("--workflow-id", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "worker":
        asyncio.run(cmd_worker(args))
    elif args.cmd == "start":
        asyncio.run(cmd_start(args))
    elif args.cmd == "approve":
        asyncio.run(cmd_approve(args))
    elif args.cmd == "replay":
        asyncio.run(cmd_replay(args))


if __name__ == "__main__":
    main()
