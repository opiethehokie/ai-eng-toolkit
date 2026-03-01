"""Compare Responses API WebSocket mode vs HTTP streaming for tool loops.

This demo intentionally keeps scope small:
- One local function tool (`lookup_order_status`)
- One repeated tool-calling workflow
- Two transport modes (`ws` and `http`)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

TOOL_NAME = "lookup_order_status"
TOOL_DESCRIPTION = "Look up status metadata for an order ID."
STATUS_SEQUENCE = ("processing", "packed", "shipped", "delivered", "exception")
MAX_TURNS_MULTIPLIER = 3
MAX_TURNS_BASE = 6


@dataclass
class DemoConfig:
    mode: str
    model: str
    tool_calls: int
    tool_latency_ms: int
    runs: int
    verbose_events: bool


@dataclass
class PendingToolCall:
    call_id: str
    name: str
    arguments_json: str


@dataclass
class RunMetrics:
    mode: str
    run_index: int
    total_elapsed_ms: float = 0.0
    model_elapsed_ms: float = 0.0
    tool_elapsed_ms: float = 0.0
    other_elapsed_ms: float = 0.0
    model_turns: int = 0
    tool_calls_executed: int = 0
    connections_opened: int = 0
    avg_model_turn_elapsed_ms: float = 0.0
    avg_end_to_end_turn_elapsed_ms: float = 0.0
    final_text_chars: int = 0
    event_count: int = 0
    errors: list[str] = field(default_factory=list)
    final_text: str = ""
    _turn_elapsed_ms: list[float] = field(default_factory=list, repr=False)

    def finalize(self) -> None:
        if self._turn_elapsed_ms:
            self.model_elapsed_ms = sum(self._turn_elapsed_ms)
            self.avg_model_turn_elapsed_ms = self.model_elapsed_ms / len(self._turn_elapsed_ms)
        if self.model_turns > 0:
            self.avg_end_to_end_turn_elapsed_ms = self.total_elapsed_ms / self.model_turns
        self.other_elapsed_ms = max(0.0, self.total_elapsed_ms - self.model_elapsed_ms - self.tool_elapsed_ms)
        self.final_text_chars = len(self.final_text)
        self.total_elapsed_ms = round(self.total_elapsed_ms, 2)
        self.model_elapsed_ms = round(self.model_elapsed_ms, 2)
        self.tool_elapsed_ms = round(self.tool_elapsed_ms, 2)
        self.other_elapsed_ms = round(self.other_elapsed_ms, 2)
        self.avg_model_turn_elapsed_ms = round(self.avg_model_turn_elapsed_ms, 2)
        self.avg_end_to_end_turn_elapsed_ms = round(self.avg_end_to_end_turn_elapsed_ms, 2)


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="Compare Responses API WebSocket mode against HTTP streaming for multi-turn tool loops.")
    parser.add_argument("--mode", choices=("ws", "http", "both"), default="both")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--tool-calls", type=int, default=4, help="Target number of tool calls in the workflow.")
    parser.add_argument("--tool-latency-ms", type=int, default=100, help="Artificial local tool latency per call.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--verbose-events", action="store_true", help="Print every event type as it arrives.")
    args = parser.parse_args()

    if args.tool_calls < 1:
        parser.error("--tool-calls must be >= 1")
    if args.tool_latency_ms < 0:
        parser.error("--tool-latency-ms must be >= 0")
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    return DemoConfig(
        mode=args.mode,
        model=args.model,
        tool_calls=args.tool_calls,
        tool_latency_ms=args.tool_latency_ms,
        runs=args.runs,
        verbose_events=args.verbose_events,
    )


def build_order_ids(tool_calls: int, run_index: int) -> list[str]:
    start = 1000 + (run_index - 1) * tool_calls
    return [f"order-{start + i}" for i in range(tool_calls)]


def build_initial_prompt(order_ids: list[str]) -> str:
    joined = ", ".join(order_ids)
    return (
        "You are running a deterministic order audit demo.\n"
        f"Order IDs to process (in order): {joined}\n"
        f"Use the `{TOOL_NAME}` tool exactly once per response turn. Do not batch calls.\n"
        "After receiving tool outputs for all order IDs, return a concise summary with one line per order."
    )


def build_tool_definition() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order identifier, e.g. order-1001"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }
    ]


def build_request_kwargs(config: DemoConfig, payload_input: Any, previous_response_id: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": 0,
        "parallel_tool_calls": False,
        "max_tool_calls": 1,
        "tools": build_tool_definition(),
        "input": payload_input,
    }
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id
    return kwargs


def lookup_order_status(order_id: str, latency_ms: int) -> dict[str, Any]:
    if latency_ms:
        time.sleep(latency_ms / 1000.0)

    numeric_portion = "".join(ch for ch in order_id if ch.isdigit())
    base = int(numeric_portion) if numeric_portion else sum(ord(c) for c in order_id)
    status = STATUS_SEQUENCE[base % len(STATUS_SEQUENCE)]
    return {
        "order_id": order_id,
        "status": status,
        "last_update_minutes_ago": (base % 43) + 3,
        "warehouse": f"WH-{(base % 5) + 1}",
    }


def parse_pending_tool_calls(event: Any) -> list[PendingToolCall]:
    if getattr(event, "type", "") != "response.output_item.done":
        return []
    item = getattr(event, "item", None)
    if item is None or getattr(item, "type", "") != "function_call":
        return []
    name = str(getattr(item, "name", ""))
    if name != TOOL_NAME:
        return []
    return [
        PendingToolCall(
            call_id=str(getattr(item, "call_id", "")),
            name=name,
            arguments_json=str(getattr(item, "arguments", "{}")),
        )
    ]


def extract_output_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", "") == "output_text":
                chunks.append(str(getattr(part, "text", "")))
    return "".join(chunks).strip()


def extract_event_error(event: Any) -> str:
    event_type = getattr(event, "type", "unknown")
    if event_type == "error":
        return str(getattr(event, "message", "unknown error"))
    response = getattr(event, "response", None)
    if response is None:
        return f"{event_type} (no response payload)"
    error_obj = getattr(response, "error", None)
    if error_obj is None:
        return f"{event_type} (no error details)"
    message = getattr(error_obj, "message", None)
    code = getattr(error_obj, "code", None)
    if message and code:
        return f"{code}: {message}"
    if message:
        return str(message)
    return str(error_obj)


def render_event_trace(mode: str, event: Any) -> None:
    event_type = getattr(event, "type", "unknown")
    if event_type == "response.output_item.done":
        item = getattr(event, "item", None)
        item_type = getattr(item, "type", "unknown")
        if item_type == "function_call":
            print(f"[{mode}] event={event_type} tool={getattr(item, 'name', '?')}")
            return
    print(f"[{mode}] event={event_type}")


def execute_tool_calls(calls: list[PendingToolCall], config: DemoConfig, metrics: RunMetrics) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for call in calls:
        try:
            arguments = json.loads(call.arguments_json)
        except json.JSONDecodeError:
            metrics.errors.append(f"invalid tool args json for call_id={call.call_id}")
            arguments = {}

        order_id = str(arguments.get("order_id", "")).strip()
        if not order_id:
            metrics.errors.append(f"missing order_id in tool args for call_id={call.call_id}")
            result: dict[str, Any] = {"ok": False, "error": "missing order_id"}
        else:
            tool_started = time.perf_counter()
            result = {"ok": True, "result": lookup_order_status(order_id, config.tool_latency_ms)}
            metrics.tool_elapsed_ms += (time.perf_counter() - tool_started) * 1000.0
            metrics.tool_calls_executed += 1

        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, separators=(",", ":")),
            }
        )
    return outputs


def run_ws_mode(client: OpenAI, config: DemoConfig, run_index: int) -> RunMetrics:
    metrics = RunMetrics(mode="ws", run_index=run_index)
    started = time.perf_counter()
    order_ids = build_order_ids(config.tool_calls, run_index)
    next_input: Any = build_initial_prompt(order_ids)
    previous_response_id: str | None = None
    max_turns = config.tool_calls * MAX_TURNS_MULTIPLIER + MAX_TURNS_BASE

    try:
        with client.responses.connect() as connection:
            metrics.connections_opened = 1

            while True:
                if metrics.model_turns >= max_turns:
                    metrics.errors.append(f"max turns reached ({max_turns})")
                    break

                metrics.model_turns += 1
                turn_started = time.perf_counter()
                request_kwargs = build_request_kwargs(config, next_input, previous_response_id)
                connection.response.create(**request_kwargs)

                pending_calls: list[PendingToolCall] = []
                turn_final_text = ""
                turn_done = False

                while not turn_done:
                    event = connection.recv()
                    metrics.event_count += 1
                    if config.verbose_events:
                        render_event_trace("ws", event)

                    pending_calls.extend(parse_pending_tool_calls(event))
                    event_type = getattr(event, "type", "")
                    if event_type == "response.completed":
                        response = getattr(event, "response", None)
                        if response is not None:
                            previous_response_id = str(getattr(response, "id", ""))
                            turn_final_text = extract_output_text(response)
                        turn_done = True
                    elif event_type in ("response.failed", "response.incomplete", "error"):
                        metrics.errors.append(f"event={event_type}: {extract_event_error(event)}")
                        turn_done = True

                metrics._turn_elapsed_ms.append((time.perf_counter() - turn_started) * 1000.0)

                if pending_calls:
                    next_input = execute_tool_calls(pending_calls, config, metrics)
                    continue

                if turn_final_text:
                    metrics.final_text = turn_final_text
                break
    except OpenAIError as exc:
        message = str(exc)
        if "openai[realtime]" in message:
            metrics.errors.append("WebSocket support missing. Install with: pip install 'openai[realtime]'")
        else:
            metrics.errors.append(f"OpenAI error: {message}")
    except Exception as exc:
        metrics.errors.append(f"unexpected ws error: {exc}")

    metrics.total_elapsed_ms = (time.perf_counter() - started) * 1000.0
    metrics.finalize()
    return metrics


def run_http_mode(client: OpenAI, config: DemoConfig, run_index: int) -> RunMetrics:
    metrics = RunMetrics(mode="http", run_index=run_index)
    started = time.perf_counter()
    order_ids = build_order_ids(config.tool_calls, run_index)
    next_input: Any = build_initial_prompt(order_ids)
    previous_response_id: str | None = None
    max_turns = config.tool_calls * MAX_TURNS_MULTIPLIER + MAX_TURNS_BASE

    try:
        while True:
            if metrics.model_turns >= max_turns:
                metrics.errors.append(f"max turns reached ({max_turns})")
                break

            metrics.model_turns += 1
            metrics.connections_opened += 1
            turn_started = time.perf_counter()
            request_kwargs = build_request_kwargs(config, next_input, previous_response_id)

            pending_calls: list[PendingToolCall] = []
            turn_final_text = ""
            turn_response_id: str | None = None

            with client.responses.stream(**request_kwargs) as stream:
                for event in stream:
                    metrics.event_count += 1
                    if config.verbose_events:
                        render_event_trace("http", event)

                    pending_calls.extend(parse_pending_tool_calls(event))
                    event_type = getattr(event, "type", "")

                    if event_type == "response.completed":
                        response = getattr(event, "response", None)
                        if response is not None:
                            turn_response_id = str(getattr(response, "id", ""))
                            turn_final_text = extract_output_text(response)
                    elif event_type in ("response.failed", "response.incomplete", "error"):
                        metrics.errors.append(f"event={event_type}: {extract_event_error(event)}")

                if turn_response_id is None:
                    final_response = stream.get_final_response()
                    turn_response_id = str(getattr(final_response, "id", ""))
                    turn_final_text = extract_output_text(final_response)

            previous_response_id = turn_response_id
            metrics._turn_elapsed_ms.append((time.perf_counter() - turn_started) * 1000.0)

            if pending_calls:
                next_input = execute_tool_calls(pending_calls, config, metrics)
                continue

            if turn_final_text:
                metrics.final_text = turn_final_text
            break
    except OpenAIError as exc:
        metrics.errors.append(f"OpenAI error: {exc}")
    except Exception as exc:
        metrics.errors.append(f"unexpected http error: {exc}")

    metrics.total_elapsed_ms = (time.perf_counter() - started) * 1000.0
    metrics.finalize()
    return metrics


def run_selected_modes(client: OpenAI, config: DemoConfig) -> list[RunMetrics]:
    selected_modes = ("ws", "http") if config.mode == "both" else (config.mode,)
    results: list[RunMetrics] = []

    for mode in selected_modes:
        for run_index in range(1, config.runs + 1):
            print(f"\n[{mode}] run {run_index}/{config.runs}")
            if mode == "ws":
                metrics = run_ws_mode(client, config, run_index)
            else:
                metrics = run_http_mode(client, config, run_index)
            print_run_summary(metrics)
            results.append(metrics)

    return results


def print_run_summary(metrics: RunMetrics) -> None:
    print(
        f"[{metrics.mode}] total_ms={metrics.total_elapsed_ms:.2f} "
        f"model_ms={metrics.model_elapsed_ms:.2f} tool_ms={metrics.tool_elapsed_ms:.2f} "
        f"other_ms={metrics.other_elapsed_ms:.2f} "
        f"turns={metrics.model_turns} tools={metrics.tool_calls_executed} "
        f"connections={metrics.connections_opened} "
        f"avg_model_turn_ms={metrics.avg_model_turn_elapsed_ms:.2f} "
        f"avg_e2e_turn_ms={metrics.avg_end_to_end_turn_elapsed_ms:.2f} "
        f"events={metrics.event_count} errors={len(metrics.errors)}"
    )
    if metrics.final_text:
        print(f"[{metrics.mode}] final: {metrics.final_text}")
    if metrics.errors:
        for err in metrics.errors:
            print(f"[{metrics.mode}] error: {err}")


def summarize_by_mode(results: list[RunMetrics]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for mode in ("ws", "http"):
        mode_runs = [r for r in results if r.mode == mode]
        if not mode_runs:
            continue
        n = len(mode_runs)
        summary[mode] = {
            "runs": float(n),
            "avg_total_elapsed_ms": round(sum(r.total_elapsed_ms for r in mode_runs) / n, 2),
            "avg_model_elapsed_ms": round(sum(r.model_elapsed_ms for r in mode_runs) / n, 2),
            "avg_tool_elapsed_ms": round(sum(r.tool_elapsed_ms for r in mode_runs) / n, 2),
            "avg_other_elapsed_ms": round(sum(r.other_elapsed_ms for r in mode_runs) / n, 2),
            "avg_model_turns": round(sum(r.model_turns for r in mode_runs) / n, 2),
            "avg_tool_calls_executed": round(sum(r.tool_calls_executed for r in mode_runs) / n, 2),
            "avg_connections_opened": round(sum(r.connections_opened for r in mode_runs) / n, 2),
            "avg_model_turn_elapsed_ms": round(sum(r.avg_model_turn_elapsed_ms for r in mode_runs) / n, 2),
            "avg_end_to_end_turn_elapsed_ms": round(sum(r.avg_end_to_end_turn_elapsed_ms for r in mode_runs) / n, 2),
            "avg_event_count": round(sum(r.event_count for r in mode_runs) / n, 2),
            "error_runs": float(sum(1 for r in mode_runs if r.errors)),
        }
    return summary


def print_comparison_table(summary: dict[str, dict[str, float]]) -> None:
    ws = summary.get("ws")
    http = summary.get("http")
    if not ws or not http:
        return

    print("\nComparison (averages across runs)")
    print(
        "mode  | total_ms | model_ms | tool_ms | other_ms | turns | tools | connections | "
        "avg_model_turn_ms | avg_e2e_turn_ms | events | error_runs"
    )
    print(
        "------|----------|----------|---------|----------|-------|-------|-------------|"
        "------------------|-----------------|--------|-----------"
    )
    for mode_name, row in (("ws", ws), ("http", http)):
        print(
            f"{mode_name:<5}| {row['avg_total_elapsed_ms']:>8.2f} | "
            f"{row['avg_model_elapsed_ms']:>8.2f} | {row['avg_tool_elapsed_ms']:>7.2f} | "
            f"{row['avg_other_elapsed_ms']:>8.2f} | {row['avg_model_turns']:>5.2f} | "
            f"{row['avg_tool_calls_executed']:>5.2f} | {row['avg_connections_opened']:>11.2f} | "
            f"{row['avg_model_turn_elapsed_ms']:>16.2f} | {row['avg_end_to_end_turn_elapsed_ms']:>15.2f} | "
            f"{row['avg_event_count']:>6.2f} | {row['error_runs']:>9.2f}"
        )

    conn_delta = http["avg_connections_opened"] - ws["avg_connections_opened"]
    model_turn_delta = http["avg_model_turn_elapsed_ms"] - ws["avg_model_turn_elapsed_ms"]
    e2e_turn_delta = http["avg_end_to_end_turn_elapsed_ms"] - ws["avg_end_to_end_turn_elapsed_ms"]
    other_delta = http["avg_other_elapsed_ms"] - ws["avg_other_elapsed_ms"]
    print(
        f"\nDelta (http - ws): connections={conn_delta:.2f}, "
        f"avg_model_turn_ms={model_turn_delta:.2f}, "
        f"avg_e2e_turn_ms={e2e_turn_delta:.2f}, "
        f"other_ms={other_delta:.2f}."
    )


def main() -> None:
    config = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required. Set it in your shell or .env and retry.")
    client = OpenAI(api_key=api_key)

    print(
        "Running demo with "
        f"mode={config.mode}, model={config.model}, tool_calls={config.tool_calls}, "
        f"tool_latency_ms={config.tool_latency_ms}, runs={config.runs}"
    )
    print("Fairness controls: identical prompt/tool schema/temperature/max_tool_calls across both modes.")

    results = run_selected_modes(client, config)
    summary = summarize_by_mode(results)

    if config.mode == "both":
        print_comparison_table(summary)


if __name__ == "__main__":
    main()
