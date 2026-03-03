"""
llmsupervisor — Independent Audit & Governance Layer for Production GenAI Systems.

Public API:
    supervisor              — singleton LLMSupervisor instance (ready to use)
    BudgetExceededError     — raised when cumulative cost exceeds the set budget
    LowConfidenceError      — raised when guard mode blocks a low-confidence execution

Phase 5 additions:
    supervisor.observe()    — passively audit pre-computed LLM outputs
    supervisor.monitor()    — decorator: wrap any function with full audit & governance
    supervisor.set_alert_mode() — True = warnings (default), False = raise exceptions

Usage:
    from llmsupervisor import supervisor

    supervisor.set_budget(0.10)
    supervisor.set_alert_mode(True)

    @supervisor.monitor(model="gpt-4o")
    def my_llm_function(prompt):
        ...
"""

from llmsupervisor.supervisor import LLMSupervisor as _LLMSupervisor
from llmsupervisor.supervisor import BudgetExceededError, LowConfidenceError

supervisor = _LLMSupervisor()

__all__ = ["supervisor", "BudgetExceededError", "LowConfidenceError"]
__version__ = "0.1.0"
