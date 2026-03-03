"""
12_fastapi_app.py — FastAPI /ask + /audit endpoints
Run: python integration-examples/12_fastapi_app.py
Then: curl http://localhost:8001/audit/status
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from pydantic import BaseModel
from llmauditor import auditor

app = FastAPI(title="LLMAuditor FastAPI Integration Test")

auditor.set_budget(10.00)
auditor.set_alert_mode(True)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    audit: dict

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    # Simulated LLM response (no real API call)
    answer = f"Simulated answer to: {request.question}"
    report = auditor.execute(
        model="gpt-4o",
        input_tokens=len(request.question.split()) * 2,
        output_tokens=len(answer.split()) * 2,
        raw_response=answer,
        input_text=request.question
    )
    return QueryResponse(answer=answer, audit=report.to_dict())

@app.get("/audit/status")
async def audit_status():
    return auditor.get_budget_status()

@app.post("/audit/evaluate")
async def run_evaluation():
    auditor.start_evaluation("FastAPI Test App", version="1.0.0")
    auditor.end_evaluation()
    report = auditor.generate_evaluation_report()
    return report.to_dict()

def main():
    """Quick self-test without starting server."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    print("=" * 60)
    print("FastAPI Integration Test (using TestClient)")
    print("=" * 60)

    # Test /ask
    r = client.post("/ask", json={"question": "What is Python?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "audit" in data
    print(f"  ✓ POST /ask → {data['answer'][:50]}...")

    # Test /audit/status
    r = client.get("/audit/status")
    assert r.status_code == 200
    status = r.json()
    assert "budget_limit" in status
    print(f"  ✓ GET /audit/status → budget: ${status['budget_limit']}")

    # Test /audit/evaluate
    r = client.post("/audit/evaluate")
    assert r.status_code == 200
    eval_data = r.json()
    assert "score" in eval_data
    print(f"  ✓ POST /audit/evaluate → level: {eval_data['score']['level']}")

    auditor.clear_history()
    print("\n✅ FASTAPI INTEGRATION TEST PASSED")

if __name__ == "__main__":
    main()
