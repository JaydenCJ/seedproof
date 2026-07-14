"""Report rendering: tables, token quoting, diff context windows, matrices."""

from __future__ import annotations

from conftest import make_record
from seedproof import build_matrix, diagnose
from seedproof.report import (
    format_table,
    matrix_to_dict,
    quote_token,
    render_diff,
    render_matrix,
    render_show,
)


def test_format_table_aligns_columns_without_trailing_whitespace():
    lines = format_table(("A", "BB"), [("longer", "x"), ("s", "yy")])
    assert lines[0] == "A       BB"
    assert lines[1] == "longer  x"
    assert all(line == line.rstrip() for line in lines)


def test_quote_token_escapes_truncates_and_marks_stream_end():
    assert quote_token("a\nb") == '"a\\nb"'  # control chars stay visible
    assert quote_token(" cat") == '" cat"'   # leading spaces stay visible
    long = quote_token("x" * 100)
    assert len(long) <= 24
    assert long.endswith('…"')
    assert quote_token(None) == "<end>"


def test_render_diff_marks_the_divergence_with_a_caret():
    a = make_record(["t0", "t1", "t2", "t3", "t4"], name="a", backend="cpu")
    b = make_record(["t0", "t1", "XX", "t3", "t4"], name="b", backend="cuda")
    lines = render_diff(diagnose(a, b), a, b, context=1)
    caret = [line for line in lines if "<- first divergence" in line]
    assert len(caret) == 1
    assert caret[0].lstrip().startswith("> 2")
    assert '"XX"' in caret[0]
    # context=1 shows exactly indices 1..3 plus the ellipsis.
    assert any(" 1  " in line for line in lines)
    assert not any('"t0"' in line for line in lines)
    assert "  ..." in lines
    # A divergence at index 0 must not underflow the window.
    a0 = make_record(["A", "y"], name="a0", backend="cpu")
    b0 = make_record(["B", "y"], name="b0", backend="cuda")
    assert any("> 0" in line for line in render_diff(diagnose(a0, b0), a0, b0))


def test_render_diff_identical_path_is_compact():
    a = make_record(["x"], name="a")
    b = make_record(["x"], name="b")
    lines = render_diff(diagnose(a, b), a, b)
    assert any("verdict: identical" in line for line in lines)
    assert not any("first divergent" in line for line in lines)
    # Counts are pluralized correctly: a one-token run is "1 token".
    assert any("(1 token)" in line for line in lines)
    assert not any("1 tokens" in line for line in lines)


def test_render_diff_shows_run_headers_with_token_counts():
    a = make_record(["x", "y"], name="baseline", backend="cpu", quant="fp32", seed=1)
    lines = render_diff(diagnose(a, a), a, a)
    assert lines[0].startswith("a: baseline")
    assert "2 tokens" in lines[0]
    single = make_record(["x"], name="tiny")
    assert render_diff(diagnose(single, single), single, single)[0].endswith("1 token")


def test_render_matrix_consistent_message():
    runs = [make_record(["x"], name=f"r{i}") for i in range(2)]
    lines = render_matrix(build_matrix(runs))
    assert any("identical token stream" in line for line in lines)


def test_render_matrix_text_and_json_agree_on_the_split():
    runs = [
        make_record(["x", "y"], name="c", backend="cpu"),
        make_record(["x", "q"], name="g", backend="cuda"),
    ]
    report = build_matrix(runs)
    text = "\n".join(render_matrix(report))
    assert "backend" in text
    assert "explains the split" in text
    assert "A vs B" in text
    data = matrix_to_dict(report)
    assert data["consistent"] is False
    assert data["classes"][0]["members"] == ["c"]
    assert data["pairwise"][0]["index"] == 1
    assert data["axes"][0]["field"] == "backend"


def test_render_show_truncates_token_list():
    record = make_record([f"t{i}" for i in range(12)], name="big")
    lines = render_show(record, limit=3)
    assert any("(9 more)" in line for line in lines)
    assert any("tokens: 12" in line for line in lines)
