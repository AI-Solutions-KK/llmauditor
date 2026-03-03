"""
report.py — ExecutionReport dataclass and CLI display logic.

Responsibilities:
- Define the ExecutionReport data structure (all execution metrics)
- Compute quality signals: confidence score, risk level, system notes
- Render a rich CLI audit panel (display())
- Delegate export to the exporter module
- Provide stable to_dict() serialisation contract

This module uses the `rich` library for terminal output.
No governance logic lives here — this is pure data + presentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

_console = Console()


# ── ExecutionReport ───────────────────────────────────────────────────────── #

@dataclass
class ExecutionReport:
    """
    Immutable record of a single LLM execution with full audit trail.

    Core fields are set at construction time by ControlEngine / LLMSupervisor.
    Quality signals (confidence, risk, notes, summary) are computed on demand.

    Phase 4.5:  ai_summary — optional AI-generated executive summary text.
    Phase 5:    warnings  — governance violations captured in alert mode.
    """

    # ── Core metrics (required at construction) ─────────────────────────── #
    execution_id:   str
    model_name:     str
    execution_time: float
    input_tokens:   int
    output_tokens:  int
    total_tokens:   int
    estimated_cost: float
    raw_response:   str

    # ── Optional fields (post-construction injection) ─────────────────────── #
    ai_summary: Optional[str] = field(default=None, compare=False, repr=False)

    # Phase 5: governance warnings (populated by observe() in alert mode)
    warnings: list[str] = field(default_factory=list, compare=False, repr=False)

    # ── Public display ────────────────────────────────────────────────────── #

    def display(self) -> None:
        """
        Render a structured CLI audit panel using rich.

        Sections:
            1. Header
            2. Model Info
            3. Usage Metrics
            4. Quality Metrics
            5. System Notes
            6. Governance Warnings  (only shown when warnings list is non-empty)
            7. AI Executive Summary (only shown when ai_summary is set)
        """
        confidence_score, confidence_label = self._compute_confidence()
        risk_level, risk_label = self._compute_risk()
        notes = self._generate_notes()
        summary = self._generate_summary()

        # ── Section 1: Header ─────────────────────────────────────────────── #
        header_text = Text(
            "LLM SUPERVISOR  ·  EXECUTION REPORT",
            style="bold white on navy_blue",
            justify="center",
        )

        # ── Section 2: Model Info ─────────────────────────────────────────── #
        model_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        model_table.add_column(style="dim cyan", no_wrap=True)
        model_table.add_column(style="white")
        model_table.add_row("Execution ID", self.execution_id)
        model_table.add_row("Model", self.model_name)
        model_table.add_row("Execution Time", f"{self.execution_time} sec")

        # ── Section 3: Usage Metrics ──────────────────────────────────────── #
        usage_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        usage_table.add_column(style="dim cyan", no_wrap=True)
        usage_table.add_column(style="white")
        usage_table.add_row("Input Tokens",  f"{self.input_tokens:,}")
        usage_table.add_row("Output Tokens", f"{self.output_tokens:,}")
        usage_table.add_row("Total Tokens",  f"[bold]{self.total_tokens:,}[/bold]")
        usage_table.add_row("Estimated Cost", f"[bold green]${self.estimated_cost:.6f} USD[/bold green]")

        # ── Section 4: Quality Metrics ────────────────────────────────────── #
        conf_color = "green" if confidence_score >= 85 else "yellow" if confidence_score >= 70 else "red"
        risk_color = "green" if risk_level == "LOW" else "yellow"

        quality_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        quality_table.add_column(style="dim cyan", no_wrap=True)
        quality_table.add_column()
        quality_table.add_row(
            "Confidence Score",
            f"[bold {conf_color}]{confidence_score}%[/bold {conf_color}]  {confidence_label}",
        )
        quality_table.add_row(
            "Risk Level",
            f"[bold {risk_color}]{risk_level}[/bold {risk_color}]  {risk_label}",
        )

        # ── Section 5: System Notes ───────────────────────────────────────── #
        notes_text = Text()
        for i, note in enumerate(notes):
            notes_text.append(f"  • {note}", style="dim white")
            if i < len(notes) - 1:
                notes_text.append("\n")

        # ── Assemble all sections ─────────────────────────────────────────── #
        from rich.columns import Columns
        from rich.console import Group

        content_parts = [
            header_text,
            "",
            "[bold dim]MODEL INFO[/bold dim]",
            model_table,
            "[bold dim]USAGE METRICS[/bold dim]",
            usage_table,
            "[bold dim]QUALITY METRICS[/bold dim]",
            quality_table,
            "[bold dim]SYSTEM NOTES[/bold dim]",
            notes_text,
        ]

        # ── Section 6: Governance Warnings (Phase 5) ──────────────────────── #
        if self.warnings:
            warnings_text = Text()
            for i, warning in enumerate(self.warnings):
                warnings_text.append(f"  ⚠  {warning}", style="bold yellow")
                if i < len(self.warnings) - 1:
                    warnings_text.append("\n")
            content_parts += [
                "",
                "[bold yellow]GOVERNANCE WARNINGS[/bold yellow]",
                warnings_text,
            ]

        # ── Section 7: AI Executive Summary (Phase 4.5) ──────────────────── #
        if self.ai_summary:
            summary_text = Text(f"\n  {self.ai_summary}\n", style="italic white")
            content_parts += [
                "",
                "[bold dim]AI EXECUTIVE SUMMARY[/bold dim]",
                summary_text,
            ]
        else:
            summary_text = Text(f"\n  {summary}\n", style="italic dim white")
            content_parts += [
                "",
                "[bold dim]AI EXECUTIVE SUMMARY[/bold dim]",
                summary_text,
            ]

        _console.print(
            Panel(
                Group(*content_parts),
                border_style="navy_blue",
                padding=(1, 2),
            )
        )

    # ── Export ────────────────────────────────────────────────────────────── #

    def export(self, fmt: str = "md", output_dir: str = ".") -> str:
        """
        Export this report to a file in the specified format.

        Args:
            fmt:        Export format. One of: 'md', 'html', 'pdf'. Default: 'md'.
            output_dir: Directory to write the output file. Default: current dir.

        Returns:
            Absolute path to the generated file.
        """
        from llmsupervisor.exporter import export as _export

        confidence_score, confidence_label = self._compute_confidence()
        risk_level, _ = self._compute_risk()
        notes = self._generate_notes()
        summary = self._generate_summary()

        quality = {
            "confidence_score": confidence_score,
            "risk_level": risk_level,
            "notes": notes,
            "summary": summary,
        }

        return _export(
            data=self.to_dict(),
            quality=quality,
            fmt=fmt,
            output_dir=output_dir,
        )

    # ── Serialisation ─────────────────────────────────────────────────────── #

    def to_dict(self) -> dict:
        """
        Return a stable serialisation of the report.

        Keys are guaranteed to remain stable across patch versions.
        Computed quality signals are included for downstream consumers.
        """
        confidence_score, confidence_label = self._compute_confidence()
        risk_level, risk_label = self._compute_risk()

        return {
            "execution_id":     self.execution_id,
            "model_name":       self.model_name,
            "execution_time":   self.execution_time,
            "input_tokens":     self.input_tokens,
            "output_tokens":    self.output_tokens,
            "total_tokens":     self.total_tokens,
            "estimated_cost":   self.estimated_cost,
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "risk_level":       risk_level,
            "risk_label":       risk_label,
            "warnings":         list(self.warnings),
        }

    # ── Private quality computers ──────────────────────────────────────────── #

    def _compute_confidence(self) -> tuple[int, str]:
        """
        Compute a confidence score (0–100) and a human-readable label.

        Heuristic:
            - Start at 100
            - Penalise for very short responses (possible truncation or refusal)
            - Penalise for very high cost (aggressive token usage)
            - Penalise for slow execution (possible timeout risk)

        Returns:
            (score: int, label: str)
        """
        score = 100

        # Penalise very short raw responses
        if len(self.raw_response) < 20:
            score -= 30
        elif len(self.raw_response) < 50:
            score -= 15

        # Penalise high cost (> $0.05 per execution)
        if self.estimated_cost > 0.05:
            score -= 20
        elif self.estimated_cost > 0.02:
            score -= 10

        # Penalise slow execution (> 10 seconds)
        if self.execution_time > 10:
            score -= 15
        elif self.execution_time > 5:
            score -= 5

        score = max(0, score)

        if score >= 85:
            label = "High — execution completed within expected parameters"
        elif score >= 70:
            label = "Medium — minor anomalies detected, review advised"
        else:
            label = "Low — significant anomalies detected, manual review required"

        return score, label

    def _compute_risk(self) -> tuple[str, str]:
        """
        Compute a risk level (LOW / MEDIUM / HIGH) and a human-readable label.

        Based on cost and token usage thresholds.

        Returns:
            (risk_level: str, risk_label: str)
        """
        if self.estimated_cost > 0.05 or self.total_tokens > 8000:
            level = "HIGH"
            label = "High usage detected — consider rate limiting or token budgets"
        elif self.estimated_cost > 0.01 or self.total_tokens > 3000:
            level = "MEDIUM"
            label = "Moderate usage — monitor cumulative cost across sessions"
        else:
            level = "LOW"
            label = "Within normal operating parameters"

        return level, label

    def _generate_notes(self) -> list[str]:
        """
        Generate a list of clinical system notes based on execution metrics.

        Notes are deterministic given the same execution data.
        """
        notes = []
        confidence_score, _ = self._compute_confidence()
        risk_level, _ = self._compute_risk()

        if self.execution_time > 10:
            notes.append(
                f"Execution time {self.execution_time}s exceeds 10s threshold — "
                f"consider async processing or model switch."
            )
        elif self.execution_time > 5:
            notes.append(
                f"Execution time {self.execution_time}s is elevated — "
                f"monitor for latency trends."
            )
        else:
            notes.append(f"Execution time {self.execution_time}s is within normal range.")

        if self.total_tokens > 8000:
            notes.append(
                f"Token usage {self.total_tokens:,} is very high — "
                f"review prompt efficiency and consider chunking."
            )
        elif self.total_tokens > 3000:
            notes.append(
                f"Token usage {self.total_tokens:,} is moderate — "
                f"track cumulative spend."
            )
        else:
            notes.append(f"Token usage {self.total_tokens:,} is efficient.")

        if self.estimated_cost > 0.05:
            notes.append(
                f"Cost ${self.estimated_cost:.6f} USD is above the $0.05 high-cost threshold."
            )
        elif self.estimated_cost > 0.01:
            notes.append(
                f"Cost ${self.estimated_cost:.6f} USD is moderate — "
                f"review if this model is cost-appropriate."
            )
        else:
            notes.append(f"Cost ${self.estimated_cost:.6f} USD is within budget norms.")

        if len(self.raw_response) < 20:
            notes.append(
                "Response is very short — possible refusal, empty output, or truncation."
            )

        if risk_level == "HIGH":
            notes.append(
                "HIGH risk execution — recommend immediate review and budget cap assessment."
            )

        return notes

    def _generate_summary(self) -> str:
        """
        Generate a rule-based executive summary string.

        If ai_summary has been injected by LLMSupervisor, that takes precedence
        and this method is not called for display — but it is always called for export.
        The ai_summary field is checked upstream in display() and export().
        """
        if self.ai_summary:
            return self.ai_summary

        confidence_score, _ = self._compute_confidence()
        risk_level, _ = self._compute_risk()

        quality = "satisfactory" if confidence_score >= 85 else "adequate" if confidence_score >= 70 else "concerning"
        risk_note = (
            "within normal operating parameters"
            if risk_level == "LOW"
            else "flagged for elevated resource usage"
            if risk_level == "MEDIUM"
            else "flagged as high-risk and requires immediate review"
        )

        return (
            f"This execution of {self.model_name} completed in {self.execution_time}s, "
            f"consuming {self.total_tokens:,} tokens at an estimated cost of "
            f"${self.estimated_cost:.6f} USD. "
            f"Overall confidence is {quality} ({confidence_score}%), "
            f"and the execution risk profile is {risk_note}."
        )
