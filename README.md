# LLMControlEngine

**AI Governance, Audit & Execution Control Framework for Generative AI Applications.**

LLMControlEngine wraps any LLM callable and provides live token tracking, cost estimation, structured audit reporting, and a governance-ready execution pipeline — without locking you into any specific AI provider.

---

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | Core Execution & Tracking | ✅ Complete |
| Phase 2 | Professional CLI Reporting (rich) | 🔲 Planned |
| Phase 3 | Audit Export (MD / HTML / PDF) | 🔲 Planned |
| Phase 4 | Governance & Control Layer | 🔲 Planned |

---

## Quick Start

```python
from llmcontrolengine import control

# Your LLM callable — any function that talks to any model.
# Must return: {"response": str, "input_tokens": int, "output_tokens": int, "model": str}
def call_openai(prompt: str) -> dict:
    # replace with your actual API call
    return {
        "response": "Merchant risk is assessed as LOW.",
        "input_tokens": 520,
        "output_tokens": 290,
        "model": "gpt-4",
    }

report = control.execute(llm=call_openai, input_data="Analyze merchant risk")
report.display()
```

**Output:**

```
====================================================
  LLMControlEngine — Execution Report
====================================================
  Execution ID   : 3f7a2c1e-...
  Model          : gpt-4
  Execution Time : 1.2431 sec
----------------------------------------------------
  Input Tokens   : 520
  Output Tokens  : 290
  Total Tokens   : 810
  Estimated Cost : $0.033000
----------------------------------------------------
  Response       : Merchant risk is assessed as LOW.
====================================================
```

---

## Import Styles

```python
# Style 1 — direct
from llmcontrolengine import control
report = control.execute(llm=my_function, input_data="text")

# Style 2 — aliased
from llmcontrolengine import control as ctrl
report = ctrl.execute(llm=my_function, input_data="text")
```

---

## LLM Callable Contract

Your `llm` function must:
- Accept a single `str` argument (the input prompt)
- Return a `dict` with exactly these keys:

```python
{
    "response":      str,   # the model output
    "input_tokens":  int,   # tokens consumed by the prompt
    "output_tokens": int,   # tokens in the completion
    "model":         str,   # model name (used for cost lookup)
}
```

---

## Supported Models (Cost Estimation)

| Model | Input (per 1K) | Output (per 1K) |
|-------|---------------|-----------------|
| gpt-4 | $0.030 | $0.060 |
| gpt-4-turbo | $0.010 | $0.030 |
| gpt-4o | $0.005 | $0.015 |
| gpt-3.5-turbo | $0.0005 | $0.0015 |
| claude-3-opus | $0.015 | $0.075 |
| claude-3-sonnet | $0.003 | $0.015 |
| claude-3-haiku | $0.00025 | $0.00125 |
| gemini-pro | $0.00025 | $0.0005 |
| *(unlisted model)* | $0.010 | $0.030 (default) |

---

## Project Structure

```
llm-control-engine/
│
├── llmcontrolengine/
│   ├── __init__.py     # Exposes singleton `control`
│   ├── control.py      # ControlEngine class — orchestration
│   ├── tracker.py      # ExecutionTracker — time & token aggregation
│   ├── cost.py         # MODEL_PRICING + calculate_cost()
│   └── report.py       # ExecutionReport dataclass + display()
│
├── tests/
├── pyproject.toml
└── README.md
```

---

## Design Principles

- **Minimal core** — zero external dependencies in Phase 1
- **Model-agnostic** — works with any callable LLM wrapper
- **Separation of concerns** — tracking, pricing, reporting are isolated modules
- **Expandable** — each phase adds capability without breaking existing API
- **Business-readable output** — designed for developers, supervisors, and compliance teams
- **Governance-first** — built to support audit trails, budget enforcement, and role control in future phases

---

## Roadmap

**Phase 2** — Rich CLI output, confidence scoring, risk classification, AI executive summary

**Phase 3** — Markdown / HTML / PDF export, digital integrity hash, audit template

**Phase 4** — Role-based limits, budget enforcement, guard mode, execution history, drift detection

---

## Contributing

This project is designed for open-source contribution. Each phase maps to a GitHub milestone with clearly scoped issues. See the [Issues](https://github.com/yourusername/llm-control-engine/issues) tab.

---

## License

MIT
