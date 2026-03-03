"""
09_export_reports.py — MD, HTML, PDF export + export_all + export_certification_all
No API key needed.
"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmauditor import auditor, export_certification_all

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports_integration_test")

def main():
    # Clean output dir
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    auditor.clear_history()

    # ── TEST 1: Per-execution export ──
    print("=" * 60)
    print("TEST 1: Per-execution export (MD, HTML, PDF)")
    print("=" * 60)
    report = auditor.execute(
        model="gpt-4o",
        input_tokens=200,
        output_tokens=100,
        raw_response="Detailed analysis of the market trends shows strong growth.",
        input_text="Analyze market trends"
    )

    md_path = report.export("md", output_dir=OUT_DIR)
    html_path = report.export("html", output_dir=OUT_DIR)
    pdf_path = report.export("pdf", output_dir=OUT_DIR)

    assert os.path.exists(md_path), f"MD not found: {md_path}"
    assert os.path.exists(html_path), f"HTML not found: {html_path}"
    assert os.path.exists(pdf_path), f"PDF not found: {pdf_path}"

    print(f"  ✓ MD:   {os.path.basename(md_path)} ({os.path.getsize(md_path)} bytes)")
    print(f"  ✓ HTML: {os.path.basename(html_path)} ({os.path.getsize(html_path)} bytes)")
    print(f"  ✓ PDF:  {os.path.basename(pdf_path)} ({os.path.getsize(pdf_path)} bytes)")

    # ── TEST 2: Certification export ──
    print("\n" + "=" * 60)
    print("TEST 2: Certification report export")
    print("=" * 60)
    auditor.clear_history()
    auditor.start_evaluation("Export Test App", version="1.0.0")
    for i in range(5):
        auditor.execute(
            model="gpt-4o", input_tokens=200, output_tokens=100,
            raw_response=f"Substantial response {i+1} for evaluation purposes."
        )
    auditor.end_evaluation()

    eval_report = auditor.generate_evaluation_report()
    cert_md = eval_report.export("md", output_dir=OUT_DIR)
    cert_html = eval_report.export("html", output_dir=OUT_DIR)
    cert_pdf = eval_report.export("pdf", output_dir=OUT_DIR)

    assert os.path.exists(cert_md)
    assert os.path.exists(cert_html)
    assert os.path.exists(cert_pdf)

    print(f"  ✓ Cert MD:   {os.path.basename(cert_md)}")
    print(f"  ✓ Cert HTML: {os.path.basename(cert_html)}")
    print(f"  ✓ Cert PDF:  {os.path.basename(cert_pdf)}")

    # ── TEST 3: export_all() ──
    print("\n" + "=" * 60)
    print("TEST 3: export_all() — all 3 formats at once")
    print("=" * 60)
    paths = eval_report.export_all(output_dir=OUT_DIR)
    assert "md" in paths
    assert "html" in paths
    assert "pdf" in paths
    for fmt, p in paths.items():
        assert os.path.exists(p), f"{fmt} not found: {p}"
        print(f"  ✓ {fmt.upper()}: {os.path.basename(p)}")

    # ── TEST 4: export_certification_all() standalone ──
    print("\n" + "=" * 60)
    print("TEST 4: export_certification_all() standalone function")
    print("=" * 60)
    paths2 = export_certification_all(eval_report, output_dir=OUT_DIR)
    assert "md" in paths2
    assert "html" in paths2
    assert "pdf" in paths2
    for fmt, p in paths2.items():
        assert os.path.exists(p)
        print(f"  ✓ {fmt.upper()}: {os.path.basename(p)}")

    # Cleanup
    auditor.clear_history()
    shutil.rmtree(OUT_DIR)
    print(f"\n  (Cleaned up {OUT_DIR})")
    print("\n✅ ALL EXPORT TESTS PASSED")

if __name__ == "__main__":
    main()
