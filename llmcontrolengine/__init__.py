"""
llmcontrolengine — AI Governance, Audit & Execution Control Framework.

Public API (Phase 1):

    from llmcontrolengine import control

    report = control.execute(llm=my_llm_function, input_data="text")
    report.display()

Only `control` is exported at this level.
Internal modules are importable directly for advanced use or testing.
"""

from llmcontrolengine.control import ControlEngine as _ControlEngine

# Singleton instance — the primary public interface of the package.
# Underscore alias prevents ControlEngine from leaking into the public namespace.
control = _ControlEngine()

__all__ = ["control"]
__version__ = "0.1.0"
