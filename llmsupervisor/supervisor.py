"""
supervisor.py — LLMSupervisor orchestration layer.

Responsibilities:
- Orchestrate a single LLM execution end-to-end (execute)
- Passively observe pre-computed results without invoking the LLM (observe)
- Provide a monitor() decorator to wrap any function with audit & governance
- Delegate time tracking to ExecutionTracker
- Delegate cost calculation to cost module
- Build and return ExecutionReport
- Enforce governance policies:
    budget limits, role-based token limits, guard mode
- Maintain in-memory execution history
- Keep this class free of display or export logic

Custom exceptions (module-level):
    BudgetExceededError   — cumulative cost would exceed the set budget limit
    LowConfidenceError    — confidence score is below the guard mode threshold
    PermissionError is used for role token limit violations (built-in)

Phase 5 additions:
    set_alert_mode(enabled)   — True = attach violations as warnings (default)
                                False = raise exceptions
    observe(...)              — passively audit existing output, no LLM call
    monitor(model, enforce)   — decorator: wraps any function with full audit
"""

import functools
import uuid
from typing import Callable, Optional

from llmsupervisor.tracker import ExecutionTracker
from llmsupervisor.cost import calculate_cost
from llmsupervisor.report import ExecutionReport


# ── Custom exceptions ─────────────────────────────────────────────────────── #

class BudgetExceededError(Exception):
    """Raised when an execution would push cumulative cost beyond the set budget limit."""


class LowConfidenceError(Exception):
    """Raised when guard mode is active and the execution confidence score is too low."""


# ── Internal constants ────────────────────────────────────────────────────── #

_REQUIRED_KEYS = {"response", "input_tokens", "output_tokens", "model"}


# ── LLMSupervisor ─────────────────────────────────────────────────────────── #

class LLMSupervisor:
    """
    Independent audit and governance layer for production GenAI systems.

    Wraps any callable LLM function, tracks execution metrics, estimates cost,
    enforces governance policies, and returns a structured ExecutionReport.
    Can also passively observe pre-computed LLM outputs without invoking the LLM.

    Governance state is session-scoped (in-memory only).
    All state resets when the process restarts.

    Core API:
        execute(llm, input_data)                  — invoke llm callable + full audit
        observe(input_data, output_data, ...)     — passive audit of existing output
        monitor(model, enforce)                   — decorator for any function

    Governance API:
        set_budget(limit_usd)                     — set maximum cumulative spend
        get_budget_status()                       — inspect current spend & remaining
        set_role(role_name, max_tokens)           — register a named role with token limit
        assign_role(role_name)                    — activate a role for executions
        guard_mode(enabled, threshold)            — block low-confidence executions
        set_alert_mode(enabled)                   — True=warnings (Phase 5 default)
        history()                                 — list completed ExecutionReport objects
        clear_history()                           — wipe in-memory execution history
        enable_ai_summary(enabled, llm_callable)  — AI-generated executive summary
    """

    def __init__(self) -> None:
        # Budget state
        self._budget_limit: Optional[float] = None
        self._budget_used: float = 0.0

        # Role state
        self._roles: dict[str, int] = {}
        self._current_role: Optional[str] = None

        # Guard mode state
        self._guard_enabled: bool = False
        self._guard_threshold: int = 75

        # Alert mode (Phase 5): True = attach warnings, False = raise exceptions
        self._alert_mode: bool = True

        # Execution history (in-memory, session only)
        self._history: list[ExecutionReport] = []

        # AI executive summary state (Phase 4.5)
        self._ai_summary_enabled: bool = False
        self._ai_summary_llm: Optional[Callable] = None

    # ── Core execution ────────────────────────────────────────────────────── #

    def execute(self, llm: Callable[[str], dict], input_data: str) -> ExecutionReport:
        """
        Execute a callable LLM function and return a tracked ExecutionReport.

        Args:
            llm:        A callable that accepts a single string (input_data)
                        and returns a dict with keys:
                            { "response": str, "input_tokens": int,
                              "output_tokens": int, "model": str }
            input_data: The prompt or input string passed to the llm callable.

        Returns:
            ExecutionReport — added to history() on success.

        Raises:
            TypeError:           If llm is not callable.
            ValueError:          If llm return value is malformed.
            BudgetExceededError: If execution cost would exceed the budget limit
                                 (only when alert_mode=False).
            PermissionError:     If token usage exceeds the active role limit
                                 (only when alert_mode=False).
            LowConfidenceError:  If guard mode is on and confidence is below threshold
                                 (only when alert_mode=False).
        """
        if not callable(llm):
            raise TypeError("llm argument must be a callable (function or object with __call__).")

        self._check_budget_headroom()

        execution_id = str(uuid.uuid4())
        tracker = ExecutionTracker()

        tracker.start()
        result = llm(input_data)
        tracker.stop()

        self._validate_result(result)

        tokens = tracker.aggregate_tokens(
            input_tokens=int(result["input_tokens"]),
            output_tokens=int(result["output_tokens"]),
        )

        cost = calculate_cost(
            model=result["model"],
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
        )

        report = ExecutionReport(
            execution_id=execution_id,
            model_name=str(result["model"]),
            execution_time=tracker.execution_time,
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            total_tokens=tokens["total_tokens"],
            estimated_cost=cost,
            raw_response=str(result["response"]),
        )

        self._apply_governance(report, tokens["total_tokens"], cost, enforce=not self._alert_mode)

        self._maybe_inject_ai_summary(report, llm)
        self._history.append(report)
        return report

    # ── Passive observation (Phase 5) ─────────────────────────────────────── #

    def observe(
        self,
        input_data: str,
        output_data: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        enforce: bool = False,
    ) -> ExecutionReport:
        """
        Passively audit a pre-computed LLM interaction.

        Unlike execute(), this method does NOT invoke any callable.
        Use it to supervise outputs that were produced externally (e.g. from
        an existing chatbot, pipeline, or third-party API response).

        Args:
            input_data:    The original prompt or input text.
            output_data:   The LLM response text to audit.
            input_tokens:  Number of input tokens consumed.
            output_tokens: Number of output tokens produced.
            model:         Model identifier (e.g. "gpt-4o").
            enforce:       If True, raise exceptions on governance violations
                           regardless of alert_mode. Default: False (attach warnings).

        Returns:
            ExecutionReport — added to history(). Violations appear in report.warnings
            unless enforce=True, in which case exceptions are raised.
        """
        execution_id = str(uuid.uuid4())

        total_tokens = input_tokens + output_tokens
        cost = calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        report = ExecutionReport(
            execution_id=execution_id,
            model_name=model,
            execution_time=0.0,   # not measured — observation only
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=cost,
            raw_response=output_data,
        )

        should_enforce = enforce or not self._alert_mode
        self._apply_governance(report, total_tokens, cost, enforce=should_enforce)

        self._history.append(report)
        return report

    # ── monitor() decorator (Phase 5) ─────────────────────────────────────── #

    def monitor(self, model: str, enforce: bool = False):
        """
        Decorator: wrap any function to automatically audit and govern its output.

        The wrapped function must return either:
          - A dict with keys: "response" (str), "input_tokens" (int),
            "output_tokens" (int), and optionally "model" (str overrides decorator arg).
          - A plain string, in which case tokens are estimated heuristically
            and the provided model is used.

        After the wrapped function returns, observe() is called automatically.
        The audit report is displayed to the terminal. The original return value
        (dict or str) is returned to the caller unchanged.

        Args:
            model:    Model name to use for cost calculation when the return
                      value is a plain string.
            enforce:  If True, raise governance exceptions instead of warnings.

        Example:
            @supervisor.monitor(model="gpt-4o")
            def ask_llm(prompt):
                return {"response": "...", "input_tokens": 100, "output_tokens": 50}
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                original_result = func(*args, **kwargs)

                # Determine input text for observe()
                input_text = ""
                if args:
                    input_text = str(args[0])
                elif kwargs:
                    input_text = str(next(iter(kwargs.values())))

                # Parse result to extract tokens and response
                if isinstance(original_result, dict) and "response" in original_result:
                    output_data    = str(original_result.get("response", ""))
                    input_tokens   = int(original_result.get("input_tokens", 0))
                    output_tokens  = int(original_result.get("output_tokens", 0))
                    used_model     = str(original_result.get("model", model))
                else:
                    # Plain string return — estimate tokens heuristically
                    output_data   = str(original_result)
                    # ~4 chars per token (rough average across models)
                    output_tokens = max(1, len(output_data) // 4)
                    input_tokens  = max(1, len(input_text) // 4)
                    used_model    = model

                report = self.observe(
                    input_data=input_text,
                    output_data=output_data,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=used_model,
                    enforce=enforce,
                )
                report.display()

                return original_result

            return wrapper
        return decorator

    # ── Alert mode API (Phase 5) ──────────────────────────────────────────── #

    def set_alert_mode(self, enabled: bool = True) -> None:
        """
        Set the governance violation response strategy.

        Args:
            enabled: True (default) — governance violations are attached as
                     warnings to the ExecutionReport and execution continues.
                     False — governance violations raise exceptions immediately
                     (BudgetExceededError, PermissionError, LowConfidenceError).

        Note: The enforce parameter on observe() and monitor() takes precedence
        over alert mode for those specific calls.
        """
        self._alert_mode = bool(enabled)

    # ── Budget API ────────────────────────────────────────────────────────── #

    def set_budget(self, limit_usd: float) -> None:
        """
        Set the maximum cumulative cost allowed for this session.

        Args:
            limit_usd: Budget ceiling in USD (e.g. 0.10 for 10 cents).

        Raises:
            ValueError: If limit_usd is not a positive number.
        """
        if not isinstance(limit_usd, (int, float)) or limit_usd <= 0:
            raise ValueError(f"Budget limit must be a positive number. Got: {limit_usd!r}")
        self._budget_limit = float(limit_usd)

    def get_budget_status(self) -> dict:
        """
        Return the current budget state for this session.

        Returns:
            dict with keys: limit_usd, used_usd, remaining_usd, executions
        """
        remaining = (
            round(self._budget_limit - self._budget_used, 6)
            if self._budget_limit is not None
            else None
        )
        return {
            "limit_usd":     self._budget_limit,
            "used_usd":      round(self._budget_used, 6),
            "remaining_usd": remaining,
            "executions":    len(self._history),
        }

    # ── Role API ──────────────────────────────────────────────────────────── #

    def set_role(self, role_name: str, max_tokens: int) -> None:
        """
        Register a named role with a maximum token limit per execution.

        Args:
            role_name:  Identifier for this role (e.g. "intern", "analyst").
            max_tokens: Maximum total tokens (input + output) allowed per execution.

        Raises:
            ValueError: If max_tokens is not a positive integer.
        """
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer. Got: {max_tokens!r}")
        self._roles[role_name] = max_tokens

    def assign_role(self, role_name: str) -> None:
        """
        Activate a registered role for all subsequent execute()/observe() calls.

        Args:
            role_name: Must be a role previously registered via set_role().

        Raises:
            KeyError: If role_name has not been registered.
        """
        if role_name not in self._roles:
            raise KeyError(
                f"Role '{role_name}' is not registered. "
                f"Register it first with set_role()."
            )
        self._current_role = role_name

    # ── Guard mode API ────────────────────────────────────────────────────── #

    def guard_mode(self, enabled: bool = True, threshold: int = 75) -> None:
        """
        Enable or disable guard mode with a minimum confidence threshold.

        When enabled and alert_mode=False (or enforce=True), raises LowConfidenceError
        if the execution's confidence score falls below the threshold.
        In alert mode (default), attaches a warning to the report instead.

        Args:
            enabled:   True to activate guard mode, False to deactivate.
            threshold: Minimum acceptable confidence percentage (0–100).

        Raises:
            ValueError: If threshold is not between 0 and 100.
        """
        if not isinstance(threshold, int) or not (0 <= threshold <= 100):
            raise ValueError(f"threshold must be an integer between 0 and 100. Got: {threshold!r}")
        self._guard_enabled = enabled
        self._guard_threshold = threshold

    # ── History API ───────────────────────────────────────────────────────── #

    def history(self) -> list[ExecutionReport]:
        """Return a shallow copy of the in-memory execution history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Wipe all in-memory execution history for this session."""
        self._history.clear()

    # ── AI Summary API (Phase 4.5) ────────────────────────────────────────── #

    def enable_ai_summary(self, enabled: bool = True, llm_callable: Optional[Callable] = None) -> None:
        """
        Enable or disable AI-generated executive summaries.

        When enabled, after each successful execute() call, a second LLM call
        generates a natural-language executive summary and injects it into the
        ExecutionReport. Fails silently — falls back to rule-based summary.

        Args:
            enabled:        True to activate AI summaries, False to deactivate.
            llm_callable:   Optional dedicated LLM callable for summary generation.
                            Must satisfy the same contract as execute()'s llm arg.
                            If None, the same callable passed to execute() is reused.
        """
        self._ai_summary_enabled = enabled
        self._ai_summary_llm = llm_callable if enabled else None

    # ── Private governance helpers ─────────────────────────────────────────── #

    def _apply_governance(
        self,
        report: ExecutionReport,
        total_tokens: int,
        cost: float,
        enforce: bool,
    ) -> None:
        """
        Apply all governance checks to a completed report.

        If enforce=True, governance failures raise exceptions immediately.
        If enforce=False, violations are appended to report.warnings and
        budget spend is committed regardless (alert-only mode).
        """
        # 1. Role token limit
        if self._current_role is not None:
            max_tokens = self._roles.get(self._current_role)
            if max_tokens is not None and total_tokens > max_tokens:
                msg = (
                    f"Role '{self._current_role}' allows a maximum of {max_tokens:,} tokens "
                    f"per execution. This execution used {total_tokens:,} tokens."
                )
                if enforce:
                    raise PermissionError(msg)
                report.warnings.append(f"[ROLE LIMIT] {msg}")

        # 2. Budget
        if self._budget_limit is not None:
            projected = self._budget_used + cost
            if projected > self._budget_limit:
                msg = (
                    f"Execution cost ${cost:.6f} would exceed remaining budget. "
                    f"Used: ${self._budget_used:.6f} / Limit: ${self._budget_limit:.6f} USD. "
                    f"Remaining: ${self._budget_limit - self._budget_used:.6f} USD."
                )
                if enforce:
                    raise BudgetExceededError(msg)
                report.warnings.append(f"[BUDGET] {msg}")
        # Always commit spend (in alert mode we continue past violations)
        self._budget_used += cost

        # 3. Guard mode confidence
        if self._guard_enabled:
            confidence_score, _ = report._compute_confidence()
            if confidence_score < self._guard_threshold:
                msg = (
                    f"Guard mode: confidence score {confidence_score}% is below "
                    f"the required threshold of {self._guard_threshold}%."
                )
                if enforce:
                    raise LowConfidenceError(msg)
                report.warnings.append(f"[GUARD MODE] {msg}")

    def _check_budget_headroom(self) -> None:
        """
        Fast-fail check before calling the LLM.

        If a budget limit is set and already fully exhausted, raise immediately
        (this check is always enforced — we never call the LLM past a zero budget).
        """
        if self._budget_limit is not None and self._budget_used >= self._budget_limit:
            raise BudgetExceededError(
                f"Budget already exhausted. "
                f"Used: ${self._budget_used:.6f} / Limit: ${self._budget_limit:.6f} USD. "
                f"No further executions are allowed in this session."
            )

    def _maybe_inject_ai_summary(self, report: ExecutionReport, execute_llm: Callable) -> None:
        """
        Attempt to generate an AI executive summary and inject it into the report.

        Uses dedicated summary LLM if set, otherwise falls back to execute_llm.
        This method never raises — all errors are silently swallowed.
        """
        if not self._ai_summary_enabled:
            return

        try:
            llm = self._ai_summary_llm or execute_llm
            confidence_score, _ = report._compute_confidence()
            risk_level, _ = report._compute_risk()

            prompt = (
                f"Write a professional 2-3 sentence executive summary for the following "
                f"AI model execution result. Be concise and business-appropriate.\n"
                f"Model: {report.model_name} | "
                f"Total tokens: {report.total_tokens:,} | "
                f"Cost: ${report.estimated_cost:.6f} USD | "
                f"Execution time: {report.execution_time} sec | "
                f"Confidence: {confidence_score}% | "
                f"Risk level: {risk_level}.\n"
                f"Respond with the summary text only, no headings or labels."
            )

            result = llm(prompt)
            if isinstance(result, dict) and "response" in result:
                summary_text = str(result["response"]).strip()
                if summary_text:
                    report.ai_summary = summary_text
        except Exception:
            pass

    @staticmethod
    def _validate_result(result: object) -> None:
        """Validate that the llm callable returned a well-formed dict."""
        if not isinstance(result, dict):
            raise ValueError(
                f"llm callable must return a dict. Got: {type(result).__name__}"
            )
        missing = _REQUIRED_KEYS - result.keys()
        if missing:
            raise ValueError(
                f"llm callable return dict is missing required keys: {sorted(missing)}"
            )
