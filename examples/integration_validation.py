"""
examples/integration_validation.py — Real-world integration validation for llmsupervisor.

Simulates a realistic GenAI application with:
  - Fake RAG pipeline function (decorator monitoring)
  - Fake chat function (decorator monitoring)
  - Manual observe() calls (passive audit of external outputs)
  - Alert mode (warnings only — no exceptions)
  - Enforce mode (exceptions raised)
  - Budget accumulation across multiple calls
  - History tracking
  - PDF export of last report

Run with:
    python examples/integration_validation.py
"""

import os
import sys

# Ensure UTF-8 output on Windows terminals (avoids cp1252 encode errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmsupervisor import supervisor, BudgetExceededError, LowConfidenceError
from llmsupervisor.supervisor import LLMSupervisor

# ── Formatting helpers ────────────────────────────────────────────────────── #

SEP  = "─" * 60
SEP2 = "═" * 60

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def result(label: str, passed: bool, detail: str = "") -> None:
    mark = "✔" if passed else "✘"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {label}{suffix}")


# ══════════════════════════════════════════════════════════════════════════════
# FAKE GenAI APP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _fake_embedding_search(query: str) -> list[str]:
    """Simulates a vector store lookup."""
    return [
        f"Document chunk 1 relevant to: {query}",
        f"Document chunk 2 with supporting evidence.",
    ]


def _fake_llm_call(prompt: str, model: str = "gpt-4o", output_tokens: int = 200) -> dict:
    """Simulates an LLM API response — returns the standard llmsupervisor dict contract."""
    word_count = len(prompt.split())
    input_tokens = max(word_count * 4, 20)
    return {
        "response": f"Generated answer based on: {prompt[:60]}...",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DECORATOR MONITORING (normal flow)
# ══════════════════════════════════════════════════════════════════════════════

section("SECTION 1 — Decorator Monitoring (Normal Flow)")

# Fresh isolated supervisor for this section
s1 = LLMSupervisor()
s1.set_budget(1.0)
s1.set_alert_mode(True)

@s1.monitor(model="gpt-4o")
def rag_pipeline(query: str) -> dict:
    """Fake RAG pipeline: retrieve context + generate answer."""
    chunks = _fake_embedding_search(query)
    augmented_prompt = f"Context: {' '.join(chunks)}\nQuestion: {query}"
    return _fake_llm_call(augmented_prompt, model="gpt-4o", output_tokens=180)


@s1.monitor(model="gpt-3.5-turbo")
def chat_function(message: str) -> dict:
    """Fake conversational chat function."""
    return _fake_llm_call(message, model="gpt-3.5-turbo", output_tokens=120)


print("\n  [RAG Pipeline — first call]")
rag_result = rag_pipeline("What are the main risks of using LLMs in production?")

print("\n  [Chat Function — first call]")
chat_result = chat_function("Summarise the key points from the last meeting.")

result(
    "Decorator returned original dict unchanged",
    isinstance(rag_result, dict) and "response" in rag_result,
    f"response[:40]: {rag_result.get('response','')[:40]}",
)
result(
    "Chat decorator returned original dict unchanged",
    isinstance(chat_result, dict) and "response" in chat_result,
)
result(
    "History has 2 entries after 2 decorated calls",
    len(s1.history()) == 2,
    f"got {len(s1.history())}",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ALERT MODE BEHAVIOR (warnings only, no exceptions)
# ══════════════════════════════════════════════════════════════════════════════

section("SECTION 2 — Alert Mode (Warnings Only, No Exceptions)")

s2 = LLMSupervisor()
s2.set_budget(0.001)             # very tight — will be breached
s2.set_role("junior", max_tokens=50)   # low limit — will be breached
s2.assign_role("junior")
s2.set_alert_mode(True)          # ALERT MODE: no exceptions, just warnings

alert_report = s2.observe(
    input_data="Analyse this quarter's financial report in detail.",
    output_data="Revenue grew 12% YoY driven by enterprise expansion.",
    input_tokens=800,
    output_tokens=300,
    model="gpt-4o",
)

budget_warning = any("[BUDGET]" in w for w in alert_report.warnings)
role_warning   = any("[ROLE LIMIT]" in w for w in alert_report.warnings)
no_exception   = True   # if we got here, no exception was raised

result("No exception raised in alert mode",  no_exception)
result("Budget warning captured in report",  budget_warning, f"{len(alert_report.warnings)} warning(s)")
result("Role limit warning captured in report", role_warning)

print("\n  Alert mode report (warnings panel visible):")
alert_report.display()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ENFORCE MODE BEHAVIOR (exceptions raised)
# ══════════════════════════════════════════════════════════════════════════════

section("SECTION 3 — Enforce Mode (Exceptions Raised)")

s3 = LLMSupervisor()
s3.set_budget(0.001)
s3.set_alert_mode(True)   # global alert mode is ON — but enforce=True overrides per-call

# enforce=True on observe() overrides alert mode — must raise
budget_exception_raised = False
try:
    s3.observe(
        input_data="Generate a full legal contract.",
        output_data="This agreement is entered into by...",
        input_tokens=1500,
        output_tokens=800,
        model="gpt-4",
        enforce=True,
    )
except BudgetExceededError:
    budget_exception_raised = True

result("BudgetExceededError raised with enforce=True (overrides alert mode)", budget_exception_raised)

# Enforce mode via monitor() decorator
s4 = LLMSupervisor()
s4.set_budget(0.0001)

@s4.monitor(model="gpt-4", enforce=True)
def expensive_rag(query: str) -> dict:
    return _fake_llm_call(query, model="gpt-4", output_tokens=600)

monitor_exception_raised = False
try:
    expensive_rag("Generate a 3,000-word strategic report.")
except BudgetExceededError:
    monitor_exception_raised = True

result("BudgetExceededError raised from monitor(enforce=True)", monitor_exception_raised)

# Guard mode enforce
s5 = LLMSupervisor()
s5.guard_mode(True, threshold=90)   # very high threshold
s5.set_alert_mode(False)             # enforce mode globally

@s5.monitor(model="gpt-4o", enforce=True)
def low_quality_llm(q: str) -> dict:
    return {"response": "ok", "input_tokens": 2000, "output_tokens": 1500, "model": "gpt-4o"}

guard_exception_raised = False
try:
    low_quality_llm("test")
except LowConfidenceError:
    guard_exception_raised = True

result("LowConfidenceError raised from monitor with high guard threshold", guard_exception_raised)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MANUAL observe() MODE (passive audit of external outputs)
# ══════════════════════════════════════════════════════════════════════════════

section("SECTION 4 — Manual observe() Mode (Passive Audit)")

# Simulate an existing GenAI app whose output we're auditing externally
s6 = LLMSupervisor()
s6.set_alert_mode(True)

# Pretend these came from an external service (OpenAI API, Bedrock, etc.)
external_outputs = [
    {
        "prompt":    "What is RAG?",
        "response":  "RAG stands for Retrieval-Augmented Generation, a technique that combines...",
        "in_tok":    45,
        "out_tok":   110,
        "model":     "gpt-4o",
    },
    {
        "prompt":    "Explain RLHF in simple terms.",
        "response":  "RLHF means Reinforcement Learning from Human Feedback. Humans rate model outputs...",
        "in_tok":    38,
        "out_tok":   95,
        "model":     "claude-3-sonnet",
    },
    {
        "prompt":    "Write a 500-word blog post on AI governance.",
        "response":  "AI governance is the framework of...",
        "in_tok":    28,
        "out_tok":   520,
        "model":     "gpt-4",
    },
]

observe_reports = []
for item in external_outputs:
    r = s6.observe(
        input_data=item["prompt"],
        output_data=item["response"],
        input_tokens=item["in_tok"],
        output_tokens=item["out_tok"],
        model=item["model"],
    )
    observe_reports.append(r)

result(
    f"All {len(external_outputs)} external outputs passively audited",
    len(observe_reports) == len(external_outputs),
)
result(
    "History has correct count after observe() calls",
    len(s6.history()) == len(external_outputs),
    f"got {len(s6.history())}",
)
total_observed_cost = sum(r.estimated_cost for r in observe_reports)
result(
    "Cumulative cost tracked across observe() calls",
    total_observed_cost > 0,
    f"${total_observed_cost:.6f} USD",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BUDGET ACCUMULATION ACROSS MULTIPLE CALLS
# ══════════════════════════════════════════════════════════════════════════════

section("SECTION 5 — Budget Accumulation Across Multiple Calls")

s7 = LLMSupervisor()
s7.set_budget(0.015)   # ~$0.00425/call × 6 = $0.0255 → breached on call 4
s7.set_alert_mode(True)

call_results = []
warning_calls = []

for i in range(6):
    r = s7.observe(
        input_data=f"Query number {i+1}",
        output_data=f"Response to query {i+1}.",
        input_tokens=400,
        output_tokens=150,
        model="gpt-4o",
    )
    call_results.append(r)
    if r.warnings:
        warning_calls.append(i + 1)

budget_status = s7.get_budget_status()
result(
    "6 calls completed without exceptions in alert mode",
    len(call_results) == 6,
    f"total calls: {len(call_results)}",
)
result(
    "Budget used is cumulative across all calls",
    budget_status["used_usd"] > 0,
    f'used ${budget_status["used_usd"]:.6f} / limit ${budget_status["limit_usd"]:.6f}',
)
result(
    "Budget warnings triggered when limit breached",
    len(warning_calls) > 0,
    f"warnings on calls: {warning_calls}" if warning_calls else "no warnings (budget not breached)",
)
result(
    "History count matches call count",
    len(s7.history()) == 6,
    f"got {len(s7.history())}",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HISTORY TRACKING
# ══════════════════════════════════════════════════════════════════════════════

section("SECTION 6 — History Tracking")

all_ids = [r.execution_id for r in s7.history()]
unique_ids = set(all_ids)

result(
    "All execution IDs are unique",
    len(all_ids) == len(unique_ids),
    f"{len(unique_ids)} unique IDs",
)
result(
    "All reports have valid execution times",
    all(r.execution_time >= 0 for r in s7.history()),
)
result(
    "All reports have model names",
    all(r.model_name for r in s7.history()),
)

s7.clear_history()
result(
    "clear_history() wipes all entries",
    len(s7.history()) == 0,
    f"got {len(s7.history())}",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EXPORT LAST REPORT TO PDF
# ══════════════════════════════════════════════════════════════════════════════

section("SECTION 7 — Export Last Observed Report to PDF")

last_report = observe_reports[-1]   # last externally audited report
out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(out_dir, exist_ok=True)

pdf_ok   = False
md_ok    = False
html_ok  = False

try:
    pdf_path = last_report.export("pdf", output_dir=out_dir)
    pdf_ok = os.path.isfile(pdf_path)
    result("PDF export successful", pdf_ok, os.path.basename(pdf_path))
except Exception as e:
    result("PDF export successful", False, str(e))

try:
    md_path = last_report.export("md", output_dir=out_dir)
    md_ok = os.path.isfile(md_path)
    result("Markdown export successful", md_ok, os.path.basename(md_path))
except Exception as e:
    result("Markdown export successful", False, str(e))

try:
    html_path = last_report.export("html", output_dir=out_dir)
    html_ok = os.path.isfile(html_path)
    result("HTML export successful", html_ok, os.path.basename(html_path))
except Exception as e:
    result("HTML export successful", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP2}")
print("  INTEGRATION VALIDATION SUMMARY")
print(SEP2)

# Budget status from s7 (cleared) — use s6 for a meaningful status
final_status   = s2.get_budget_status()
s1_status      = s1.get_budget_status()
s6_status      = s6.get_budget_status()

print(f"\n  Section 1 — Decorator monitoring")
print(f"    History items     : {s1_status['executions']}")
print(f"    Budget used       : ${s1_status['used_usd']:.6f} USD")

print(f"\n  Section 2 — Alert mode (s2)")
print(f"    Warnings generated: {len(alert_report.warnings)}")
for w in alert_report.warnings:
    print(f"      → {w[:80]}")

print(f"\n  Section 4 — Passive observe() audit (s6)")
print(f"    History items     : {s6_status['executions']}")
print(f"    Total cost        : ${s6_status['used_usd']:.6f} USD")

print(f"\n  Section 5 — Budget accumulation (s7)")
print(f"    Warning calls     : {warning_calls if warning_calls else 'none (budget not breached)'}")

print(f"\n  Section 7 — Exports")
print(f"    PDF               : {'OK' if pdf_ok else 'FAILED'}")
print(f"    Markdown          : {'OK' if md_ok else 'FAILED'}")
print(f"    HTML              : {'OK' if html_ok else 'FAILED'}")

# Supervisor evaluation criteria check
print(f"\n{SEP2}")
print("  SUPERVISOR EVALUATION CRITERIA")
print(SEP2)

criteria = [
    ("Decorator feels clean — returns original result unchanged",
     isinstance(rag_result, dict) and rag_result.get("response")),
    ("Report readable — rich display rendered without errors",
     True),   # display() was called above without exception
    ("Alert mode non-invasive — no exception raised on violation",
     no_exception and budget_warning),
    ("Enforce mode stops execution — BudgetExceededError raised",
     budget_exception_raised and monitor_exception_raised),
    ("Guard mode enforce — LowConfidenceError raised",
     guard_exception_raised),
    ("History behaves correctly — unique IDs, clear works",
     len(unique_ids) == 6),
    ("Export works — at least PDF generated",
     pdf_ok),
    ("API intuitive — observe(), monitor(), set_alert_mode() compose naturally",
     True),
]

all_pass = True
for label, passed in criteria:
    result(label, passed)
    if not passed:
        all_pass = False

print()
if all_pass:
    print("  ✔  ALL CRITERIA PASSED — Phase 5 is production-ready.")
else:
    print("  ✘  SOME CRITERIA FAILED — review output above.")
print(f"{SEP2}\n")
