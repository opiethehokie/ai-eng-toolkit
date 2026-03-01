import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections.abc import Callable

    import marimo as mo

    try:
        from litellm import completion
    except Exception:  # pragma: no cover - import guard for notebook UX
        completion = None
    return Callable, completion, mo


@app.cell
def _(mo):
    intro_md = "\n".join(
        [
            "# LiteLLM Gateway Demo (Retries + Fallbacks)",
            "",
            "This Marimo app shows a minimal gateway pattern for balancing cost and quality.",
            "",
            "- Primary model: lower cost, lower quality",
            "- Fallback model: higher cost, higher quality",
            "- Retry policy for transient failures",
            "- Optional quality escalation when output quality is low",
            "",
            "This notebook uses LiteLLM Router for retries/fallbacks and fault injection to make those paths deterministic.",
        ]
    )
    mo.md(intro_md)
    return


@app.cell
def _(mo):
    scenarios_md = "\n".join(
        [
            "## Suggested Scenarios",
            "",
            "| scenario | fault injection | fail first N primary calls | max retries | fallback | quality escalation | threshold | expected behavior |",
            "|---|---|---:|---:|---|---|---:|---|",
            "| baseline_cost_first | on | 0 | 2 | on | off | 0.80 | primary succeeds with lowest cost |",
            "| retry_success | on | 1 | 2 | on | off | 0.80 | primary fails once, then succeeds on retry |",
            "| retry_exhaustion_fallback | on | 3 | 1 | on | off | 0.80 | primary retries exhaust, fallback serves request |",
            "| quality_escalation | on | 0 | 1 | on | on | 0.80 | primary succeeds, then escalates for quality |",
        ]
    )
    mo.md(scenarios_md)
    return


@app.cell
def _(mo):
    max_retries = mo.ui.slider(0, 3, value=2, step=1, label="Max retries per model")
    enable_fault_injection = mo.ui.checkbox(value=True, label="Enable fault injection")
    fail_first_n = mo.ui.slider(0, 3, value=0, step=1, label="Fail first N primary calls")
    primary_model_id = mo.ui.text(
        value="gpt-4.1-mini",
        label="Primary model ID",
        full_width=True,
    )
    fallback_model_id = mo.ui.text(
        value="gpt-4.1",
        label="Fallback model ID",
        full_width=True,
    )
    enable_fallback = mo.ui.checkbox(value=True, label="Enable fallback model")
    enable_quality_escalation = mo.ui.checkbox(
        value=False,
        label="Escalate to quality model when quality score is low",
    )
    quality_threshold = mo.ui.slider(0.0, 1.0, value=0.8, step=0.05, label="Quality threshold")
    prompt = mo.ui.text_area(
        value="Give me a concise explanation of how retries and fallbacks differ in LLM gateways.",
        label="Prompt",
        rows=5,
        full_width=True,
    )
    run_btn = mo.ui.run_button(label="Run gateway call")
    run_scenarios_btn = mo.ui.run_button(label="Run all scenarios")

    mo.vstack(
        [
            mo.hstack([max_retries, enable_fault_injection, fail_first_n]),
            mo.hstack([primary_model_id, fallback_model_id]),
            mo.hstack([enable_fallback, enable_quality_escalation, quality_threshold]),
            prompt,
            mo.hstack([run_btn, run_scenarios_btn]),
        ]
    )
    return (
        enable_fallback,
        enable_fault_injection,
        enable_quality_escalation,
        fail_first_n,
        fallback_model_id,
        max_retries,
        primary_model_id,
        prompt,
        quality_threshold,
        run_btn,
        run_scenarios_btn,
    )


@app.cell
def _():
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class GatewayResult:
        final_model: str
        content: str
        quality_score: float
        estimated_cost_usd: float
        attempts: list[dict[str, Any]]

    @dataclass
    class ModelConfig:
        litellm_model: str
        prompt_token_cost: float
        completion_token_cost: float

    model_catalog = {
        "cheap_primary": ModelConfig(
            litellm_model="gpt-4.1-mini",
            prompt_token_cost=0.00000015,
            completion_token_cost=0.00000060,
        ),
        "quality_fallback": ModelConfig(
            litellm_model="gpt-4.1",
            prompt_token_cost=0.00000200,
            completion_token_cost=0.00000800,
        ),
    }
    return Any, GatewayResult, model_catalog


@app.function
def event_log_markdown(title: str, event_rows: list[dict[str, str]]) -> str:
    header = (
        "| scenario | attempt | model | retry | status | latency_ms | quality | cost_usd | error |\n|---|---:|---|---:|---|---:|---:|---:|---|"
    )
    lines = [
        f"| {r['scenario']} | {r['attempt']} | {r['model']} | {r['retry']} | {r['status']} | {r['latency_ms']} | {r['quality']} | {r['cost_usd']} | {r['error']} |"
        for r in event_rows
    ]
    if not lines:
        lines = ["| - | - | - | - | - | - | - | - | - |"]
    table_md = "\n".join([header, *lines])
    return "\n".join([title, "", table_md])


@app.cell
def _(Any, Callable, GatewayResult):
    from contextlib import contextmanager
    import time

    import litellm
    from litellm import Router

    class DeterministicFailureInjector:
        def __init__(self, fail_first_n: int):
            self.fail_first_n = fail_first_n
            self.calls_by_model: dict[str, int] = {}

    class LiteLLMGateway:
        def __init__(
            self,
            model_order: list[str],
            max_retries: int,
            quality_threshold: float,
            completion_fn: Callable | None,
            enable_fault_injection: bool,
            fail_first_n: int,
        ):
            self.model_order = model_order
            self.max_retries = max_retries
            self.quality_threshold = quality_threshold
            self.completion_fn = completion_fn
            self.enable_fault_injection = enable_fault_injection
            self.failure_injector = DeterministicFailureInjector(fail_first_n=fail_first_n)

        def _estimate_usage(self, prompt: str, answer: str) -> tuple[int, int]:
            prompt_tokens = max(1, len(prompt.split()) * 2)
            completion_tokens = max(1, len(answer.split()) * 2)
            return prompt_tokens, completion_tokens

        def _is_retryable_error(self, exc: Exception) -> bool:
            text = f"{type(exc).__name__} {exc}".lower()
            retryable_markers = (
                "ratelimit",
                "rate limit",
                "timeout",
                "timed out",
                "connection",
                "temporar",
                "service unavailable",
                "api connection",
                "internalservererror",
                "429",
                "408",
                "500",
                "502",
                "503",
                "504",
                "injected transient",
            )
            return any(marker in text for marker in retryable_markers)

        def _alias_from_model_id(self, model_id: str, primary_model_id: str, fallback_model_id: str) -> str:
            model_id_l = model_id.lower().strip()
            if model_id_l == "primary":
                return "cheap_primary"
            if model_id_l == "fallback":
                return "quality_fallback"

            # Normalize provider prefixes like "openai/gpt-4.1-mini" -> "gpt-4.1-mini"
            normalized = model_id_l.split("/")[-1]
            primary_norm = primary_model_id.lower().strip().split("/")[-1]
            fallback_norm = fallback_model_id.lower().strip().split("/")[-1]

            if normalized == primary_norm:
                return "cheap_primary"
            if normalized == fallback_norm:
                return "quality_fallback"

            if primary_norm in normalized and fallback_norm not in normalized:
                return "cheap_primary"
            if fallback_norm in normalized and primary_norm not in normalized:
                return "quality_fallback"
            return "cheap_primary"

        @contextmanager
        def _patch_litellm_completion(
            self,
            primary_model_id: str,
            fallback_model_id: str,
            prompt: str,
            model_catalog: dict[str, Any],
            attempts: list[dict[str, Any]],
        ):
            original_completion = litellm.completion

            def wrapped_completion(*args, **kwargs):
                model_id = str(kwargs.get("model") or "")
                model_alias = self._alias_from_model_id(
                    model_id=model_id,
                    primary_model_id=primary_model_id,
                    fallback_model_id=fallback_model_id,
                )
                retry_idx = self.failure_injector.calls_by_model.get(model_alias, 0)
                self.failure_injector.calls_by_model[model_alias] = retry_idx + 1
                start = time.perf_counter()

                try:
                    if self.enable_fault_injection and model_alias == "cheap_primary" and retry_idx < self.failure_injector.fail_first_n:
                        raise litellm.RateLimitError(
                            message="Injected transient 429/rate limit",
                            llm_provider="openai",
                            model=model_id or "unknown",
                        )

                    response = original_completion(*args, **kwargs)
                    content = response.choices[0].message.content or ""
                    if not isinstance(content, str):
                        content = str(content)
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    p_tokens, c_tokens = self._estimate_usage(prompt, content)
                    cfg = model_catalog[model_alias]
                    call_cost = (p_tokens * cfg.prompt_token_cost) + (c_tokens * cfg.completion_token_cost)
                    quality_score = 0.72 if model_alias == "cheap_primary" else 0.90
                    attempts.append(
                        {
                            "model_alias": model_alias,
                            "retry": retry_idx,
                            "status": "success",
                            "latency_ms": latency_ms,
                            "quality_score": round(quality_score, 3),
                            "call_cost_usd": round(call_cost, 6),
                        }
                    )
                    return response
                except Exception as exc:
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    attempts.append(
                        {
                            "model_alias": model_alias,
                            "retry": retry_idx,
                            "status": "error_retryable" if self._is_retryable_error(exc) else "error_terminal",
                            "latency_ms": latency_ms,
                            "error": str(exc),
                        }
                    )
                    raise

            litellm.completion = wrapped_completion
            try:
                yield
            finally:
                litellm.completion = original_completion

        def call(
            self,
            prompt: str,
            model_catalog: dict[str, Any],
            enable_quality_escalation: bool,
            enable_fallback: bool,
        ) -> GatewayResult:
            attempts: list[dict[str, Any]] = []
            if self.completion_fn is None:
                raise RuntimeError("litellm is not installed. Run: uv sync")

            primary_model_id = model_catalog["cheap_primary"].litellm_model
            fallback_model_id = model_catalog["quality_fallback"].litellm_model
            router_model_list = [{"model_name": "primary", "litellm_params": {"model": primary_model_id}}]
            if enable_fallback:
                router_model_list.append({"model_name": "fallback", "litellm_params": {"model": fallback_model_id}})

            router = Router(
                model_list=router_model_list,
                num_retries=self.max_retries,
                fallbacks=[{"primary": ["fallback"]}] if enable_fallback else [],
                timeout=30,
                set_verbose=False,
            )

            try:
                with self._patch_litellm_completion(
                    primary_model_id=primary_model_id,
                    fallback_model_id=fallback_model_id,
                    prompt=prompt,
                    model_catalog=model_catalog,
                    attempts=attempts,
                ):
                    response = router.completion(
                        model="primary",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
            except Exception:
                raise RuntimeError(f"All models failed. Attempts: {attempts}")

            content = response.choices[0].message.content or ""
            if not isinstance(content, str):
                content = str(content)
            response_model = str(getattr(response, "model", ""))
            final_model = self._alias_from_model_id(
                model_id=response_model,
                primary_model_id=primary_model_id,
                fallback_model_id=fallback_model_id,
            )
            final_quality = 0.72 if final_model == "cheap_primary" else 0.90

            if enable_quality_escalation and enable_fallback and final_model == "cheap_primary" and final_quality < self.quality_threshold:
                attempts.append(
                    {
                        "model_alias": "cheap_primary",
                        "retry": "-",
                        "status": "quality_escalation",
                        "latency_ms": "-",
                        "quality_score": round(final_quality, 3),
                        "call_cost_usd": "-",
                    }
                )
                with self._patch_litellm_completion(
                    primary_model_id=primary_model_id,
                    fallback_model_id=fallback_model_id,
                    prompt=prompt,
                    model_catalog=model_catalog,
                    attempts=attempts,
                ):
                    response = router.completion(
                        model="fallback",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                content = response.choices[0].message.content or ""
                if not isinstance(content, str):
                    content = str(content)
                final_model = "quality_fallback"
                final_quality = 0.90

            estimated_cost = 0.0
            for logged_attempt in attempts:
                if isinstance(logged_attempt.get("call_cost_usd"), float):
                    estimated_cost += logged_attempt["call_cost_usd"]

            return GatewayResult(
                final_model=final_model,
                content=content,
                quality_score=final_quality,
                estimated_cost_usd=estimated_cost,
                attempts=attempts,
            )

    return (LiteLLMGateway,)


@app.cell
def _(
    LiteLLMGateway,
    completion,
    enable_fallback,
    enable_fault_injection,
    enable_quality_escalation,
    fail_first_n,
    fallback_model_id,
    max_retries,
    mo,
    model_catalog,
    primary_model_id,
    prompt,
    quality_threshold,
    run_btn,
):
    mo.stop(not run_btn.value)
    mo.stop(not prompt.value.strip(), mo.md("Prompt is empty. Add a prompt and run again."))

    single_model_order = ["cheap_primary", "quality_fallback"] if enable_fallback.value else ["cheap_primary"]
    single_runtime_model_catalog = {
        **model_catalog,
        "cheap_primary": model_catalog["cheap_primary"].__class__(
            litellm_model=primary_model_id.value.strip() or model_catalog["cheap_primary"].litellm_model,
            prompt_token_cost=model_catalog["cheap_primary"].prompt_token_cost,
            completion_token_cost=model_catalog["cheap_primary"].completion_token_cost,
        ),
        "quality_fallback": model_catalog["quality_fallback"].__class__(
            litellm_model=fallback_model_id.value.strip() or model_catalog["quality_fallback"].litellm_model,
            prompt_token_cost=model_catalog["quality_fallback"].prompt_token_cost,
            completion_token_cost=model_catalog["quality_fallback"].completion_token_cost,
        ),
    }

    single_gateway = LiteLLMGateway(
        model_order=single_model_order,
        max_retries=max_retries.value,
        quality_threshold=quality_threshold.value,
        completion_fn=completion,
        enable_fault_injection=enable_fault_injection.value,
        fail_first_n=fail_first_n.value,
    )

    result = single_gateway.call(
        prompt=prompt.value.strip(),
        model_catalog=single_runtime_model_catalog,
        enable_quality_escalation=enable_quality_escalation.value,
        enable_fallback=enable_fallback.value,
    )

    single_event_rows = []
    for single_attempt_idx, single_attempt in enumerate(result.attempts):
        single_event_rows.append(
            {
                "scenario": "single_call",
                "attempt": str(single_attempt_idx + 1),
                "model": single_attempt.get("model_alias", "-"),
                "retry": str(single_attempt.get("retry", "-")),
                "status": single_attempt.get("status", "-"),
                "latency_ms": str(single_attempt.get("latency_ms", "-")),
                "quality": str(single_attempt.get("quality_score", "-")),
                "cost_usd": str(single_attempt.get("call_cost_usd", "-")),
                "error": str(single_attempt.get("error", "-")),
            }
        )

    mo.md(event_log_markdown("## Consolidated Event Log (Single Call)", single_event_rows))
    return


@app.cell
def _(
    LiteLLMGateway,
    completion,
    fallback_model_id,
    mo,
    model_catalog,
    primary_model_id,
    prompt,
    run_scenarios_btn,
):
    mo.stop(not run_scenarios_btn.value)
    mo.stop(not prompt.value.strip(), mo.md("Prompt is empty. Add a prompt and run again."))

    scenarios = [
        {
            "name": "baseline_cost_first",
            "fail_first_n": 0,
            "max_retries": 2,
            "enable_fallback": True,
            "enable_quality_escalation": False,
            "quality_threshold": 0.80,
        },
        {
            "name": "retry_success",
            "fail_first_n": 1,
            "max_retries": 2,
            "enable_fallback": True,
            "enable_quality_escalation": False,
            "quality_threshold": 0.80,
        },
        {
            "name": "retry_exhaustion_fallback",
            "fail_first_n": 3,
            "max_retries": 1,
            "enable_fallback": True,
            "enable_quality_escalation": False,
            "quality_threshold": 0.80,
        },
        {
            "name": "quality_escalation",
            "fail_first_n": 0,
            "max_retries": 1,
            "enable_fallback": True,
            "enable_quality_escalation": True,
            "quality_threshold": 0.80,
        },
    ]
    scenario_runtime_model_catalog = {
        **model_catalog,
        "cheap_primary": model_catalog["cheap_primary"].__class__(
            litellm_model=primary_model_id.value.strip() or model_catalog["cheap_primary"].litellm_model,
            prompt_token_cost=model_catalog["cheap_primary"].prompt_token_cost,
            completion_token_cost=model_catalog["cheap_primary"].completion_token_cost,
        ),
        "quality_fallback": model_catalog["quality_fallback"].__class__(
            litellm_model=fallback_model_id.value.strip() or model_catalog["quality_fallback"].litellm_model,
            prompt_token_cost=model_catalog["quality_fallback"].prompt_token_cost,
            completion_token_cost=model_catalog["quality_fallback"].completion_token_cost,
        ),
    }

    event_rows = []
    for scenario in scenarios:
        scenario_model_order = ["cheap_primary", "quality_fallback"] if scenario["enable_fallback"] else ["cheap_primary"]
        scenario_gateway = LiteLLMGateway(
            model_order=scenario_model_order,
            max_retries=scenario["max_retries"],
            quality_threshold=scenario["quality_threshold"],
            completion_fn=completion,
            enable_fault_injection=True,
            fail_first_n=scenario["fail_first_n"],
        )

        try:
            scenario_result = scenario_gateway.call(
                prompt=prompt.value.strip(),
                model_catalog=scenario_runtime_model_catalog,
                enable_quality_escalation=scenario["enable_quality_escalation"],
                enable_fallback=scenario["enable_fallback"],
            )
            for scenario_attempt_idx, scenario_attempt in enumerate(scenario_result.attempts):
                event_rows.append(
                    {
                        "scenario": scenario["name"],
                        "attempt": str(scenario_attempt_idx + 1),
                        "model": scenario_attempt.get("model_alias", "-"),
                        "retry": str(scenario_attempt.get("retry", "-")),
                        "status": scenario_attempt.get("status", "-"),
                        "latency_ms": str(scenario_attempt.get("latency_ms", "-")),
                        "quality": str(scenario_attempt.get("quality_score", "-")),
                        "cost_usd": str(scenario_attempt.get("call_cost_usd", "-")),
                        "error": str(scenario_attempt.get("error", "-")),
                    }
                )
        except Exception as exc:
            event_rows.append(
                {
                    "scenario": scenario["name"],
                    "attempt": "-",
                    "model": "-",
                    "retry": "-",
                    "status": "failed",
                    "latency_ms": "-",
                    "quality": "-",
                    "cost_usd": "-",
                    "error": str(exc)[:80],
                }
            )

    mo.md(event_log_markdown("## Consolidated Event Log (All Scenarios)", event_rows))
    return


if __name__ == "__main__":
    app.run()
