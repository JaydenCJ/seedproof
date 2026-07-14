"""Human-readable rendering of diagnoses, matrices, and records.

Everything here returns lists of plain-text lines — no printing, no color
codes — so the CLI stays a thin shell and every report is unit-testable by
string comparison.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

from .forensics import Diagnosis
from .matrix import (
    RELATION_CORRELATES,
    RELATION_EXPLAINS,
    MatrixReport,
)
from .record import RunRecord

#: Token cell wider than this gets truncated with an ellipsis.
MAX_TOKEN_CELL = 24


def pluralize(count: int, noun: str) -> str:
    """``pluralize(1, "token")`` -> ``1 token``; ``pluralize(2, ...)`` -> ``2 tokens``."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


def quote_token(text: Optional[str], limit: int = MAX_TOKEN_CELL) -> str:
    """Render a token for display: JSON-quoted, controls escaped, truncated."""
    if text is None:
        return "<end>"
    quoted = json.dumps(text, ensure_ascii=False)
    if len(quoted) > limit:
        quoted = quoted[: limit - 2] + '…"'
    return quoted


def format_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Left-aligned column layout with two-space gutters, no trailing blanks."""
    all_rows = [list(header)] + [list(row) for row in rows]
    widths = [max(len(row[col]) for row in all_rows) for col in range(len(header))]
    lines = []
    for row in all_rows:
        cells = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append("  ".join(cells).rstrip())
    return lines


def run_header(record: RunRecord, label: str) -> str:
    """One-line run summary used at the top of a diff."""
    return (
        f"{label}: {record.name or '<unnamed>'}  {record.config.summary()}  "
        f"{pluralize(len(record.tokens), 'token')}"
    )


# -- diff ---------------------------------------------------------------------


def render_diff(
    diagnosis: Diagnosis,
    a: RunRecord,
    b: RunRecord,
    context: int = 3,
) -> List[str]:
    """The full text report for ``seedproof diff``."""
    lines = [run_header(a, "a"), run_header(b, "b"), ""]
    div = diagnosis.divergence
    if not div.diverged:
        lines.append(
            f"verdict: {diagnosis.verdict} (confidence: {diagnosis.confidence})"
        )
        lines.append(f"  {diagnosis.summary}")
        for item in diagnosis.evidence:
            lines.append(f"  - [{item.label}] {item.detail}")
        return lines
    if div.kind == "token":
        lines.append(f"first divergent token: index {div.index}  (basis: {div.basis})")
    else:
        lines.append(
            f"streams match for {pluralize(div.common_len, 'token')}, then one ends "
            f"(basis: {div.basis})"
        )
    lines.append("")
    lines.extend(_context_window(a, b, div.index if div.index is not None else 0, context))
    lines.append("")
    lines.append(f"verdict: {diagnosis.verdict} (confidence: {diagnosis.confidence})")
    lines.append(f"  {diagnosis.summary}")
    if diagnosis.evidence:
        lines.append("evidence:")
        for item in diagnosis.evidence:
            lines.append(f"  - [{item.label}] {item.detail}")
    return lines


def _context_window(
    a: RunRecord, b: RunRecord, index: int, context: int
) -> List[str]:
    """Side-by-side token columns around the divergence, caret on the flip."""
    start = max(0, index - context)
    end = min(max(len(a.tokens), len(b.tokens)), index + context + 1)
    rows: List[Tuple[str, str, str, str, str]] = []
    for i in range(start, end):
        text_a = a.tokens[i].text if i < len(a.tokens) else None
        text_b = b.tokens[i].text if i < len(b.tokens) else None
        marker = ">" if i == index else ""
        note = "<- first divergence" if i == index else ""
        rows.append((marker, str(i), quote_token(text_a), quote_token(text_b), note))
    width_idx = max(len(row[1]) for row in rows)
    width_a = max(len(row[2]) for row in rows)
    width_b = max(len(row[3]) for row in rows)
    lines = []
    for marker, idx, cell_a, cell_b, note in rows:
        line = (
            f"  {marker:1} {idx.rjust(width_idx)}  "
            f"{cell_a.ljust(width_a)}  {cell_b.ljust(width_b)}  {note}"
        ).rstrip()
        lines.append(line)
    if end < max(len(a.tokens), len(b.tokens)):
        lines.append("  ...")
    return lines


# -- matrix -------------------------------------------------------------------


_RELATION_TEXT = {
    RELATION_EXPLAINS: "explains the split",
    RELATION_CORRELATES: "consistent with the split",
}


def render_matrix(report: MatrixReport) -> List[str]:
    lines = [
        f"prompt: {report.prompt_sha256[:12]}  basis: {report.basis}  "
        f"runs: {len(report.records)}  classes: {len(report.classes)}",
        "",
    ]
    rows = []
    for cls in report.classes:
        rows.append(
            (
                cls.label,
                str(len(cls.records)),
                str(len(cls.representative.tokens)),
                cls.fingerprint,
                ", ".join(cls.members),
            )
        )
    lines.extend(format_table(("CLASS", "RUNS", "TOKENS", "STREAM", "MEMBERS"), rows))
    if report.consistent:
        lines.append("")
        lines.append("all runs produced the identical token stream")
        return lines
    if report.axes:
        lines.append("")
        lines.append("varying config axes:")
        for axis in report.axes:
            relation = _RELATION_TEXT.get(axis.relation, "does not explain the split")
            mapping = "; ".join(
                f"{value} -> {'/'.join(classes)}"
                for value, classes in sorted(axis.value_classes.items())
            )
            lines.append(f"  {axis.field:<12} {relation:<27} ({mapping})")
    if report.combined_axes:
        lines.append(
            "  combined: "
            + " + ".join(report.combined_axes)
            + " together explain the split"
        )
    if report.pairwise:
        lines.append("")
        lines.append("first divergence between classes:")
        for pair in report.pairwise:
            if pair.kind == "length":
                detail = f"token {pair.index} (one stream ends)"
            else:
                detail = (
                    f"token {pair.index}  "
                    f"{quote_token(pair.a_text)} vs {quote_token(pair.b_text)}"
                )
            lines.append(f"  {pair.label_a} vs {pair.label_b}  {detail}")
    return lines


def matrix_to_dict(report: MatrixReport) -> dict:
    """JSON-friendly view (used by ``seedproof matrix --json``)."""
    return {
        "basis": report.basis,
        "prompt_sha256": report.prompt_sha256,
        "runs": len(report.records),
        "consistent": report.consistent,
        "classes": [
            {
                "label": cls.label,
                "fingerprint": cls.fingerprint,
                "members": cls.members,
                "tokens": len(cls.representative.tokens),
            }
            for cls in report.classes
        ],
        "axes": [
            {
                "field": axis.field,
                "relation": axis.relation,
                "value_classes": axis.value_classes,
            }
            for axis in report.axes
        ],
        "combined_axes": list(report.combined_axes) if report.combined_axes else None,
        "pairwise": [
            {
                "a": pair.label_a,
                "b": pair.label_b,
                "kind": pair.kind,
                "index": pair.index,
                "a_token": pair.a_text,
                "b_token": pair.b_text,
            }
            for pair in report.pairwise
        ],
    }


# -- show ---------------------------------------------------------------------


def render_show(record: RunRecord, limit: int = 8) -> List[str]:
    config = record.config
    prompt = record.prompt if len(record.prompt) <= 60 else record.prompt[:57] + "..."
    lines = [
        f"record: {record.name or '<unnamed>'}",
        f"prompt: {json.dumps(prompt, ensure_ascii=False)}  "
        f"(sha256 {record.prompt_sha256()[:12]})",
        f"config: {config.summary()}",
        f"  model={config.model or '?'}  fingerprint={config.fingerprint()}",
        f"tokens: {len(record.tokens)}"
        + ("  (with logprobs)" if record.has_logprobs() else ""),
        f"stream: {record.stream_fingerprint('text')} (text basis)"
        + (
            f"  {record.stream_fingerprint('id')} (id basis)"
            if record.ids() is not None
            else ""
        ),
    ]
    if record.created_at:
        lines.insert(1, f"created: {record.created_at}")
    if record.notes:
        lines.append(f"notes: {record.notes}")
    lines.append("")
    for i, token in enumerate(record.tokens[:limit]):
        lp = f"  logprob={token.logprob:.4f}" if token.logprob is not None else ""
        token_id = f"  id={token.id}" if token.id is not None else ""
        lines.append(f"  {i:>4}  {quote_token(token.text)}{token_id}{lp}")
    if len(record.tokens) > limit:
        lines.append(f"  ...  ({len(record.tokens) - limit} more)")
    return lines
