"""
control.py — ControlEngine orchestration layer.

Responsibilities:
- Orchestrate a single LLM execution end-to-end
- Delegate time tracking to ExecutionTracker
- Delegate cost calculation to cost module
- Build and return ExecutionReport
- Validate llm callable contract
- Keep this class free of display or export logic
"""

import uuid
from typing import Callable

from llmcontrolengine.tracker import ExecutionTracker
from llmcontrolengine.cost import calculate_cost
from llmcontrolengine.report import ExecutionReport


_REQUIRED_KEYS = {"response", "input_tokens", "output_tokens", "model"}


class ControlEngine:
    """
    Core execution orchestrator for LLMControlEngine.

    Wraps any callable LLM function, tracks execution metrics,
    estimates cost, and returns a structured ExecutionReport.

    Future phases will extend this class with:
    - Budget enforcement (Phase 4)
    - Role-based permission checks (Phase 4)
    - Guard mode (Phase 4)
    - Execution history (Phase 4)
    """

    def execute(self, llm: Callable[[str], dict], input_data: str) -> ExecutionReport:
        """
        Execute a callable LLM function and return a tracked ExecutionReport.

        Args:
            llm:        A callable that accepts a single string (input_data)
                        and returns a dict with the following required keys:
                            {
                                "response":      str,
                                "input_tokens":  int,
                                "output_tokens": int,
                                "model":         str
                            }
            input_data: The prompt or input string passed to the llm callable.

        Returns:
            ExecutionReport containing execution metrics, token usage,
            estimated cost, and raw response.

        Raises:
            TypeError:  If llm is not callable.
            ValueError: If llm return value is not a dict or is missing required keys.
        """
        if not callable(llm):
            raise TypeError("llm argument must be a callable (function or object with __call__).")

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

        return ExecutionReport(
            execution_id=execution_id,
            model_name=str(result["model"]),
            execution_time=tracker.execution_time,
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            total_tokens=tokens["total_tokens"],
            estimated_cost=cost,
            raw_response=str(result["response"]),
        )

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
