# Integration Examples

Minimal, runnable examples that validate every integration pattern documented in `DOCUMENTATION.md`.

## Setup

```bash
# From project root
pip install llmauditor openai groq langchain-openai flask fastapi uvicorn streamlit python-dotenv
```

Add your API keys to `.env` in the project root:
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

## Files

| File | Tests | Live API? |
|------|-------|-----------|
| `01_quickstart.py` | execute, observe, monitor, display, to_dict | No |
| `02_openai_live.py` | OpenAI SDK + auditor.execute | Yes (OpenAI) |
| `03_openai_decorator.py` | @monitor decorator with OpenAI | Yes (OpenAI) |
| `04_groq_live.py` | Groq SDK + custom pricing | Yes (Groq) |
| `05_langchain_openai.py` | LangChain ChatOpenAI + audit | Yes (OpenAI) |
| `06_budget_governance.py` | set_budget, guard_mode, alert_mode, role | No |
| `07_evaluation_certification.py` | Full evaluation session + certification | No |
| `08_hallucination.py` | Rule-based + AI judge hallucination | Yes (OpenAI) |
| `09_export_reports.py` | MD, HTML, PDF export + export_all | No |
| `10_custom_pricing.py` | set_pricing_table, unpriced models | No |
| `11_multi_agent.py` | Multi-step agentic pipeline + budget | No |
| `12_fastapi_app.py` | FastAPI /ask + /audit endpoints | No (server) |
| `13_flask_app.py` | Flask /ask + /audit endpoints | No (server) |
| `14_streamlit_app.py` | Streamlit chat with audit sidebar | No (app) |
| `15_error_safety.py` | Error isolation, fallbacks, warnings | No |
| `16_full_endtoend.py` | Complete pipeline: config → eval → cert → export | No |

## Run All Offline Tests

```bash
python run_all.py
```

## Run Individual

```bash
python integration-examples/01_quickstart.py
python integration-examples/02_openai_live.py      # needs OPENAI_API_KEY
python integration-examples/12_fastapi_app.py       # starts server on :8001
```
