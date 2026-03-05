"""
auditor.py — Core LLMAuditor orchestrator.

Central class that coordinates:
  - Per-execution audit (execute / observe / monitor)
  - Hallucination detection (rule-based + optional AI judge)
  - Governance enforcement (budget, guard mode, alert mode, roles)
  - Evaluation sessions (start / end / generate report)
  - Enterprise customisation (pricing, thresholds, weights)

All features from the original LLMSupervisor are preserved and enhanced.
"""

from __future__ import annotations

import functools
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from llmauditor.cost import calculate_cost, set_pricing_table as _set_pricing
from llmauditor.evaluation import (
    EvaluationMetrics,
    EvaluationReport,
    EvaluationSession,
    aggregate_metrics,
)
from llmauditor.hallucination import HallucinationDetector
from llmauditor.report import ExecutionReport
from llmauditor.scoring import ScoringEngine
from llmauditor.suggestions import SuggestionEngine
from llmauditor.tracker import ExecutionTracker


# ── Custom exceptions ─────────────────────────────────────────────────────── #

class BudgetExceededError(RuntimeError):
    """Raised when cumulative cost exceeds the configured budget limit."""


class LowConfidenceError(RuntimeError):
    """Raised when guard mode blocks a low-confidence execution."""


# ── Main class ─────────────────────────────────────────────────────────────── #

class LLMAuditor:
    """
    Execution-Based GenAI Application Evaluation & Certification Framework.

    Wraps around any LLM integration to provide per-execution auditing with
    cost tracking, token usage monitoring, hallucination detection, governance
    enforcement, and certification scoring for comprehensive evaluation.

    Attributes
    ----------
    _history : List[ExecutionReport]
        List of all execution reports recorded by this auditor instance.
    _cumulative_cost : float
        Running total of estimated costs across all executions.
    _budget_limit : Optional[float]
        Maximum allowed cumulative cost in USD (None = no limit).
    _guard_threshold : Optional[int]
        Minimum confidence score required for execution (None = disabled).
    _alert_mode : bool
        If True, governance violations trigger warnings instead of exceptions.
    _role : Optional[str]
        Current role context for governance enforcement.
    _ai_summary_enabled : bool
        Whether to generate AI summaries for execution reports.
    _ai_summary_fn : Optional[Callable]
        Custom function for generating AI summaries.
    _hallucination_detector : HallucinationDetector
        Instance for detecting hallucination risks in LLM outputs.
    _eval_session : Optional[EvaluationSession]
        Current evaluation session if one is active.
    _scoring_engine : ScoringEngine
        Engine for computing certification scores.
    _suggestion_engine : SuggestionEngine
        Engine for generating improvement suggestions.

    Examples
    --------
    Basic usage with decorator:

    >>> from llmauditor import auditor
    >>> @auditor.monitor(model="gpt-4o")
    ... def my_llm_call(prompt: str) -> dict:
    ...     # Your LLM integration here
    ...     return {"response": "...", "input_tokens": 100, "output_tokens": 50}

    Manual execution recording:

    >>> report = auditor.execute(
    ...     model="gpt-4o-mini",
    ...     input_tokens=100,
    ...     output_tokens=50,
    ...     raw_response="Sample response",
    ...     input_text="Sample prompt"
    ... )
    >>> report.display()

    Evaluation session:

    >>> auditor.start_evaluation("My App", version="1.0.0")
    >>> # ... run multiple executions ...
    >>> auditor.end_evaluation()
    >>> report = auditor.generate_evaluation_report()
    >>> report.export("pdf", "./reports")
    """

    def __init__(self) -> None:
        # ── Per-execution state ───────────────────────────────────────────── #
        self._history: list[ExecutionReport] = []
        self._cumulative_cost: float = 0.0

        # ── Governance ────────────────────────────────────────────────────── #
        self._budget_limit: Optional[float] = None
        self._guard_threshold: Optional[int] = None
        self._alert_mode: bool = False
        self._role: Optional[str] = None

        # ── AI summary ────────────────────────────────────────────────────── #
        self._ai_summary_enabled: bool = False
        self._ai_summary_fn: Optional[Callable] = None

        # ── Hallucination detection ───────────────────────────────────────── #
        self._hallucination_detector = HallucinationDetector()

        # ── Evaluation session ────────────────────────────────────────────── #
        self._eval_session: Optional[EvaluationSession] = None
        self._scoring_engine = ScoringEngine()
        self._suggestion_engine = SuggestionEngine()

    # ═══════════════════════════════════════════════════════════════════════════
    #  GOVERNANCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════

    def set_budget(self, max_cost_usd: float) -> None:
        """
        Set a cumulative cost budget limit in USD.

        Once the total cost across all executions exceeds this limit,
        subsequent executions will either raise BudgetExceededError
        (normal mode) or generate warnings (alert mode).

        Parameters
        ----------
        max_cost_usd : float
            Maximum allowed cumulative cost in USD.
            Must be a positive number.

        Notes
        -----
        Budget enforcement behavior depends on the current mode:
        - Normal mode: Raises BudgetExceededError when exceeded
        - Alert mode: Logs warning and continues execution

        The budget applies to the cumulative cost of all executions
        recorded by this auditor instance since initialization or
        since the last call to clear_history().

        Examples
        --------
        Set a $10 budget:

        >>> auditor.set_budget(10.0)
        >>> # Now all executions will be monitored against this budget
        """
        self._budget_limit = max_cost_usd

    def get_budget_status(self) -> dict[str, Any]:
        """Return current budget tracking state."""
        return {
            "budget_limit": self._budget_limit,
            "cumulative_cost": round(self._cumulative_cost, 6),
            "remaining": round(
                (self._budget_limit or 0) - self._cumulative_cost, 6
            ) if self._budget_limit is not None else None,
            "executions": len(self._history),
        }

    def guard_mode(self, confidence_threshold: int = 80) -> None:
        """
        Enable guard mode with minimum confidence threshold for execution approval.

        Executions with confidence scores below this threshold will either
        raise LowConfidenceError (normal mode) or generate warnings
        (alert mode).

        Parameters
        ----------
        confidence_threshold : int, default=80
            Minimum confidence score (0-100) required for execution approval.
            Confidence scores are computed based on response quality,
            hallucination risk, and other factors.

        Notes
        -----
        Guard mode enforcement behavior:
        - Normal mode: Raises LowConfidenceError for low-confidence executions
        - Alert mode: Adds [GUARD MODE] warning and allows execution to continue

        Confidence score ranges:
        - 85+ : High confidence
        - 70-84: Medium confidence
        - Below 70: Low confidence

        Examples
        --------
        Enable guard mode with 85% confidence requirement:

        >>> auditor.guard_mode(85)
        >>> # Now executions below 85% confidence will be blocked/warned
        """
        self._guard_threshold = confidence_threshold

    def set_alert_mode(self, enabled: bool = True) -> None:
        """
        Toggle alert mode for governance enforcement.

        When enabled, governance violations (budget exceeded, guard mode
        triggered) produce warnings on the report instead of raising
        exceptions. This allows execution to continue while still
        flagging policy violations for review.

        Parameters
        ----------
        enabled : bool, default=True
            If True, enables alert mode (warnings instead of exceptions).
            If False, disables alert mode (exceptions raised on violations).

        Notes
        -----
        Alert mode affects the following governance features:
        - Budget enforcement: Warns instead of raising BudgetExceededError
        - Guard mode: Warns instead of raising LowConfidenceError

        This is useful for monitoring and compliance tracking without
        disrupting production workflows.

        Examples
        --------
        Enable alert mode for production monitoring:

        >>> auditor.set_alert_mode(True)
        >>> auditor.set_budget(5.0)
        >>> # Budget violations will now warn instead of crash
        """
        self._alert_mode = enabled

    def set_role(self, role: str) -> None:
        """
        Set the current role context for governance enforcement.

        Role-based governance allows different execution policies
        based on the user or system context. This can be used to
        implement different budget limits, guard thresholds, or
        other policy variations.

        Parameters
        ----------
        role : str
            Role identifier (e.g., "developer", "production", "admin").
            The role context is stored and may influence governance
            decisions and reporting.

        Notes
        -----
        Role context is used for:
        - Governance policy differentiation
        - Audit trail attribution  
        - Compliance reporting

        The role is included in execution reports for tracking
        and audit purposes.

        Examples
        --------
        Set role for different environments:

        >>> auditor.set_role("production")
        >>> # Role context now applied to all executions
        """
        self._role = role

    def assign_role(self, role: str) -> None:
        """
        Alias for set_role() method.

        Parameters
        ----------
        role : str
            Role identifier to assign to this auditor instance.

        See Also
        --------
        set_role : Main method for setting role context.
        """
        self.set_role(role)

    # ═══════════════════════════════════════════════════════════════════════════
    #  AI SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    def enable_ai_summary(
        self,
        enabled: bool,
        llm_callable: Optional[Callable] = None,
    ) -> None:
        """
        Toggle AI-generated executive summaries for each execution.

        If no llm_callable is provided, the rule-based summary is used.
        """
        self._ai_summary_enabled = enabled
        if llm_callable is not None:
            self._ai_summary_fn = llm_callable

    # ═══════════════════════════════════════════════════════════════════════════
    #  CORE EXECUTION METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def execute(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        raw_response: str,
        execution_time: Optional[float] = None,
        input_text: str = "",
    ) -> ExecutionReport:
        """
        Execute an audit record for an LLM interaction.

        This function records metadata about a completed LLM call and generates
        a comprehensive execution report including token usage, estimated cost,
        confidence score, hallucination signals, and governance checks.

        Parameters
        ----------
        model : str
            Name of the LLM model used (e.g., "gpt-4o-mini", "claude-3-sonnet").
        input_tokens : int
            Number of tokens sent to the model as input/prompt.
        output_tokens : int
            Number of tokens returned by the model in the response.
        raw_response : str
            Raw text response returned by the LLM.
        execution_time : Optional[float], default=None
            Execution time in seconds. If None, defaults to 0.0.
        input_text : str, default=""
            Original prompt or user request sent to the model.
            Used for hallucination analysis and quality assessment.

        Returns
        -------
        ExecutionReport
            Structured audit report containing evaluation metrics,
            cost estimation, governance checks, and quality assessment.

        Raises
        ------
        BudgetExceededError
            If cumulative cost exceeds the configured budget limit
            (only in normal mode, not alert mode).
        LowConfidenceError
            If confidence score is below guard threshold
            (only in normal mode, not alert mode).

        Notes
        -----
        This method is for manual recording of completed LLM executions.
        The LLM call should have already happened. For automatic monitoring,
        use the `monitor` decorator instead.

        The report is not automatically displayed. Call `report.display()`
        if you want to see the audit panel.

        Examples
        --------
        Record a completed OpenAI API call:

        >>> report = auditor.execute(
        ...     model="gpt-4o-mini",
        ...     input_tokens=100,
        ...     output_tokens=50,
        ...     raw_response="The weather is sunny today.",
        ...     execution_time=1.2,
        ...     input_text="What's the weather like?"
        ... )
        >>> report.display()
        """
        try:
            return self._build_report(
                model=model,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                raw_response=str(raw_response or ""),
                execution_time=float(execution_time or 0.0),
                input_text=str(input_text or ""),
            )
        except (BudgetExceededError, LowConfidenceError):
            raise  # governance exceptions propagate
        except Exception as exc:
            # Fallback: return a minimal report so the host app never crashes
            return self._fallback_report(model, raw_response, exc)

    def observe(
        self,
        model: str,
        input_text: str,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
        execution_time: float = 0.0,
    ) -> ExecutionReport:
        """
        Observe and audit an already-completed LLM execution.

        Similar to execute() but takes explicit input/output text pairs and
        automatically displays the audit panel. This method is designed for
        observing executions that have already been completed.

        Parameters
        ----------
        model : str
            Name of the LLM model used (e.g., "gpt-4o", "claude-3-haiku").
        input_text : str
            Original prompt or user request sent to the model.
        output_text : str
            Text response returned by the LLM.
        input_tokens : int
            Number of tokens sent to the model as input.
        output_tokens : int
            Number of tokens returned by the model.
        execution_time : float, default=0.0
            Execution time in seconds.

        Returns
        -------
        ExecutionReport
            Structured audit report with hallucination analysis,
            cost estimation, and quality metrics.

        Raises
        ------
        BudgetExceededError
            If cumulative cost exceeds the configured budget limit.
        LowConfidenceError
            If confidence score is below guard threshold.

        Notes
        -----
        This method automatically calls `display()` on the generated report,
        showing the audit panel in the console. Use `execute()` if you want
        to control when the report is displayed.

        Examples
        --------
        Observe a completed conversation:

        >>> report = auditor.observe(
        ...     model="gpt-4o",
        ...     input_text="Explain quantum computing",
        ...     output_text="Quantum computing uses quantum mechanics...",
        ...     input_tokens=150,
        ...     output_tokens=300,
        ...     execution_time=2.1
        ... )
        """
        try:
            report = self._build_report(
                model=model,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                raw_response=str(output_text or ""),
                execution_time=float(execution_time or 0.0),
                input_text=str(input_text or ""),
            )
            report.display()
            return report
        except (BudgetExceededError, LowConfidenceError):
            raise
        except Exception as exc:
            return self._fallback_report(model, output_text, exc)

    def monitor(self, model: str, **monitor_kwargs: Any) -> Callable[[Callable], Callable]:
        """
        Decorator that wraps an LLM-calling function for automatic auditing.

        The decorated function should return a dictionary with "response",
        "input_tokens", and "output_tokens" keys, or return a string
        (tokens default to 0). Measures execution time, runs hallucination
        detection, applies governance, displays the report, and returns
        the original result to the caller.

        Parameters
        ----------
        model : str
            Name of the LLM model used (e.g., "gpt-4o-mini", "claude-3-sonnet").
        **monitor_kwargs : Any
            Additional keyword arguments for future extensibility.
            Currently unused but preserved for compatibility.

        Returns
        -------
        Callable[[Callable], Callable]
            Decorator function that wraps the target function.

        Notes
        -----
        The decorated function should return either:
        - A dict with keys: "response" (str), "input_tokens" (int), "output_tokens" (int)
        - A string (tokens will default to 0)

        The decorator automatically:
        - Measures execution time
        - Extracts input text from first positional argument or "prompt" keyword
        - Runs hallucination detection and quality assessment
        - Applies governance rules (budget, guard mode)
        - Displays the audit report
        - Returns the original function result unchanged

        If the auditor encounters an error, it silently falls back to calling
        the original function without auditing to ensure the host application
        never crashes due to auditing issues.

        Examples
        --------
        Decorate an OpenAI API calling function:

        >>> @auditor.monitor(model="gpt-4o-mini")
        ... def call_openai(prompt: str) -> dict:
        ...     # Your OpenAI API call here
        ...     response = openai.completions.create(...)
        ...     return {
        ...         "response": response.choices[0].text,
        ...         "input_tokens": response.usage.prompt_tokens,
        ...         "output_tokens": response.usage.completion_tokens
        ...     }

        Simple string return function:

        >>> @auditor.monitor(model="local-llm")
        ... def simple_llm(prompt: str) -> str:
        ...     return "Generated response"
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    tracker = ExecutionTracker()
                    tracker.start()
                    result = func(*args, **kwargs)
                    tracker.stop()

                    # ── Parse result ──────────────────────────────────────── #
                    if isinstance(result, dict):
                        response = str(result.get("response", ""))
                        in_tok = int(result.get("input_tokens", 0))
                        out_tok = int(result.get("output_tokens", 0))
                    else:
                        response = str(result) if result is not None else ""
                        in_tok = 0
                        out_tok = 0

                    # ── Derive input text from first positional arg ───────── #
                    input_text = ""
                    if args:
                        input_text = str(args[0])
                    elif "prompt" in kwargs:
                        input_text = str(kwargs["prompt"])

                    # ── Build, display, return ────────────────────────────── #
                    report = self._build_report(
                        model=model,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        raw_response=response,
                        execution_time=tracker.execution_time,
                        input_text=input_text,
                    )
                    report.display()
                    return result
                except (BudgetExceededError, LowConfidenceError):
                    raise  # governance exceptions propagate
                except Exception:
                    # Auditor error must never break the decorated function
                    # Re-call the original function without auditing
                    return func(*args, **kwargs)

            return wrapper
        return decorator

    # ═══════════════════════════════════════════════════════════════════════════
    #  HISTORY
    # ═══════════════════════════════════════════════════════════════════════════

    def history(self) -> List[ExecutionReport]:
        """
        Return all recorded execution reports.

        Returns
        -------
        List[ExecutionReport]
            List of all execution reports recorded by this auditor instance
            since initialization or since the last call to clear_history().
            Reports are returned in chronological order.

        Notes
        -----
        The returned list is a copy, so modifications to it will not
        affect the internal history state.

        Examples
        --------
        Review execution history:

        >>> reports = auditor.history()
        >>> print(f"Total executions: {len(reports)}")
        >>> for report in reports[-5:]:  # Show last 5
        ...     print(f"{report.model_name}: ${report.estimated_cost:.4f}")
        """
        return list(self._history)

    def clear_history(self) -> None:
        """
        Clear execution history and reset cumulative cost tracking.

        This method removes all execution reports from the internal
        history and resets the cumulative cost counter to zero.
        Budget limits and other configuration remain unchanged.

        Notes
        -----
        This operation:
        - Clears all execution reports from history
        - Resets cumulative cost to $0.00
        - Preserves all configuration (budget, guard mode, etc.)
        - Does not affect active evaluation sessions

        Use this method to start fresh cost tracking or to
        manage memory usage in long-running applications.

        Examples
        --------
        Reset for new tracking period:

        >>> auditor.clear_history()
        >>> print(f"Executions: {len(auditor.history())}")
        0
        """
        self._history.clear()
        self._cumulative_cost = 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    #  EVALUATION SESSION (NEW)
    # ═══════════════════════════════════════════════════════════════════════════

    def start_evaluation(self, app_name: str, version: str = "1.0.0",
                          mode: str = "offline") -> None:
        """
        Begin an evaluation session.

        All executions recorded between start_evaluation() and end_evaluation()
        are aggregated into the session for certification scoring.

        Args:
            app_name: Name of the application under evaluation.
            version:  Application version string.
            mode:     "simulated" (stub responses) or "live" (real API calls).
        """
        try:
            self._eval_session = EvaluationSession(
                app_name=str(app_name or "Unknown"),
                version=str(version or "1.0.0"),
                start_index=len(self._history),
                mode=str(mode or "simulated"),
            )
        except Exception:
            self._eval_session = EvaluationSession(
                app_name="Unknown", version="1.0.0",
                start_index=len(self._history),
                mode="simulated",
            )

    def end_evaluation(self) -> None:
        """
        End the current evaluation session.

        Marks the session boundary so that generate_evaluation_report()
        knows which executions to aggregate.
        """
        if self._eval_session is None:
            raise RuntimeError("No active evaluation session to end.")
        try:
            self._eval_session.end(end_index=len(self._history))
        except Exception:
            self._eval_session.end_time = __import__("datetime").datetime.now()
            self._eval_session.end_index = len(self._history)

    def generate_evaluation_report(self) -> EvaluationReport:
        """
        Generate the full evaluation certification report.

        Aggregates all executions from the evaluation session, computes
        statistical metrics, runs the scoring engine, and generates
        improvement suggestions.

        Returns:
            EvaluationReport with certification score and suggestions.

        Raises:
            RuntimeError: If no evaluation session has been started/ended.
        """
        if self._eval_session is None:
            raise RuntimeError("No evaluation session. Call start_evaluation() first.")

        try:
            start = self._eval_session.start_index
            end = self._eval_session.end_index
            if end is None:
                end = len(self._history)

            reports = self._history[start:end]

            # Aggregate metrics
            metrics = aggregate_metrics(reports)

            # Score
            score = self._scoring_engine.score(metrics.to_dict())

            # Suggestions
            suggestions = self._suggestion_engine.generate(
                metrics=metrics.to_dict(),
                subscores=score.subscores,
            )

            return EvaluationReport(
                session=self._eval_session,
                metrics=metrics,
                score=score,
                suggestions=suggestions,
                execution_reports=reports,
            )
        except Exception as exc:
            # Provide a minimal report on internal error so host app never crashes
            from llmauditor.evaluation import EvaluationMetrics
            from llmauditor.scoring import CertificationScore

            start = self._eval_session.start_index
            end = self._eval_session.end_index or len(self._history)
            reports = self._history[start:end]
            metrics = EvaluationMetrics(total_runs=len(reports))
            score = CertificationScore(
                overall=0.0, level="Fail", level_emoji="\u274c",
                subscores={}, weights={}, breakdown={},
            )
            return EvaluationReport(
                session=self._eval_session,
                metrics=metrics,
                score=score,
                suggestions=[],
                execution_reports=reports,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    #  ENTERPRISE CONFIGURATION (NEW)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_certification_thresholds(
        self,
        weights: Optional[dict[str, float]] = None,
        levels: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Customise certification scoring parameters.

        Args:
            weights: Subscore weights (must sum to 1.0).
                     Keys: stability, factual_reliability, governance_compliance,
                           cost_predictability, risk_profile
            levels:  Certification level thresholds.
                     Keys: Platinum, Gold, Silver, Conditional Pass
        """
        if weights:
            self._scoring_engine.set_weights(weights)
        if levels:
            self._scoring_engine.set_thresholds(levels)

    def set_hallucination_model(
        self,
        llm_callable: Callable,
        model: str = "gpt-4o",
    ) -> None:
        """
        Configure a secondary AI model for enhanced hallucination detection.

        The callable should accept a single string prompt and return
        either a string or a dict with a "response" key.
        """
        self._hallucination_detector.set_ai_judge(llm_callable, model)

    def set_pricing_table(
        self,
        custom_pricing: dict[str, dict[str, float]],
    ) -> None:
        """
        Override or extend the model pricing table.

        Example:
            auditor.set_pricing_table({
                "my-custom-model": {"input": 0.001, "output": 0.002}
            })
        """
        _set_pricing(custom_pricing)

    # ═══════════════════════════════════════════════════════════════════════════
    #  INTERNAL: Report builder
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_report(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        raw_response: str,
        execution_time: float,
        input_text: str,
    ) -> ExecutionReport:
        """
        Central report construction pipeline.

        Steps:
            1. Create ExecutionReport with core metrics
            2. Run hallucination detection
            3. Generate AI summary (if enabled)
            4. Apply governance checks (budget, guard mode)
            5. Append to history
        """
        exec_id = str(uuid.uuid4())
        cost = calculate_cost(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        # ── 1. Create report ──────────────────────────────────────────────── #
        report = ExecutionReport(
            execution_id=exec_id,
            model_name=model,
            execution_time=execution_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            raw_response=raw_response,
        )

        # ── 2. Hallucination analysis ─────────────────────────────────────── #
        try:
            hal = self._hallucination_detector.analyze(
                input_text=input_text or "",
                output_text=raw_response,
            )
            report.hallucination = hal
        except Exception:
            pass  # Hallucination detection failure is non-fatal

        # ── 3. AI summary ─────────────────────────────────────────────────── #
        if self._ai_summary_enabled and self._ai_summary_fn:
            try:
                summary_result = self._ai_summary_fn(
                    f"Provide a brief executive summary of this AI execution:\n"
                    f"Model: {model}\n"
                    f"Response: {raw_response[:500]}"
                )
                report.ai_summary = (
                    summary_result if isinstance(summary_result, str)
                    else str(summary_result)
                )
            except Exception:
                pass  # AI summary failure is non-fatal

        # ── 4. Governance checks ──────────────────────────────────────────── #
        # Always accumulate cost and record the execution first
        self._cumulative_cost += cost
        self._history.append(report)

        # Budget check
        if self._budget_limit is not None:
            if self._cumulative_cost > self._budget_limit:
                msg = (
                    f"[BUDGET] Cumulative cost ${self._cumulative_cost:.6f} "
                    f"exceeds budget ${self._budget_limit:.6f}"
                )
                if self._alert_mode:
                    report.warnings.append(msg)
                else:
                    raise BudgetExceededError(
                        f"Budget exceeded: ${self._cumulative_cost:.6f} > "
                        f"${self._budget_limit:.6f}"
                    )

        # Guard mode check
        if self._guard_threshold is not None:
            confidence, _ = report._compute_confidence()
            if confidence < self._guard_threshold:
                msg = (
                    f"[GUARD MODE] Confidence {confidence}% below "
                    f"threshold {self._guard_threshold}%"
                )
                if self._alert_mode:
                    report.warnings.append(msg)
                else:
                    raise LowConfidenceError(
                        f"Confidence {confidence}% below guard threshold "
                        f"{self._guard_threshold}%"
                    )

        return report

    # ── Fallback report (error safety) ─────────────────────────────────────── #

    def _fallback_report(
        self, model: Any, raw_response: Any, exc: Exception,
    ) -> ExecutionReport:
        """Create a minimal valid report when an internal error occurs."""
        report = ExecutionReport(
            execution_id=str(uuid.uuid4()),
            model_name=str(model or "unknown"),
            execution_time=0.0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost=0.0,
            raw_response=str(raw_response or "")[:500],
        )
        report.warnings.append(f"[AUDITOR ERROR] {type(exc).__name__}: {exc}")
        self._history.append(report)
        return report
