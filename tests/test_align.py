"""Alignment engine: first divergence, resync, prefix drift, tie analysis."""

from __future__ import annotations

import pytest

from conftest import make_record
from seedproof import (
    AlignError,
    analyze_tie,
    choose_basis,
    find_resync,
    first_divergence,
    prefix_drift,
)


def test_identical_streams_have_no_divergence():
    a = make_record(["x", "y", "z"], name="a")
    b = make_record(["x", "y", "z"], name="b")
    div = first_divergence(a, b)
    assert div.kind == "identical"
    assert div.index is None
    assert not div.diverged
    assert div.common_len == 3
    # Two empty streams are identical too, not a crash or a length case.
    empty = first_divergence(make_record([], name="e1"), make_record([], name="e2"))
    assert empty.kind == "identical"


def test_first_divergent_token_index_and_tokens():
    a = make_record(["x", "y", "z"], name="a")
    b = make_record(["x", "q", "z"], name="b")
    div = first_divergence(a, b)
    assert div.kind == "token"
    assert div.index == 1
    assert div.a_token.text == "y"
    assert div.b_token.text == "q"
    assert div.common_len == 1


def test_strict_prefix_is_a_length_divergence():
    a = make_record(["x", "y", "z"], name="long")
    b = make_record(["x", "y"], name="short")
    div = first_divergence(a, b)
    assert div.kind == "length"
    assert div.index == 2
    assert div.a_token.text == "z"  # the unmatched token on the longer side
    assert div.b_token is None


def test_auto_basis_prefers_ids_and_falls_back_to_text():
    with_ids = make_record(["x"], name="a")
    also_ids = make_record(["x"], name="b")
    no_ids = make_record(["x"], name="c", with_ids=False)
    assert choose_basis(with_ids, also_ids, "auto") == "id"
    assert choose_basis(with_ids, no_ids, "auto") == "text"


def test_basis_errors_are_actionable():
    a = make_record(["x"], name="a")
    b = make_record(["x"], name="b", with_ids=False)
    with pytest.raises(AlignError, match="--basis text"):
        choose_basis(a, b, "id")
    with pytest.raises(AlignError, match="unknown basis"):
        choose_basis(a, a, "bytes")


def test_id_basis_catches_same_text_with_different_ids():
    # Two vocabs can render different ids as the same text; the id basis
    # must see through that while the text basis cannot.
    a = make_record(["x", "y"], name="a")
    b = make_record(["x", "y"], name="b")
    b.tokens[1].id = a.tokens[1].id + 1
    assert first_divergence(a, b, "text").kind == "identical"
    div = first_divergence(a, b, "id")
    assert div.kind == "token"
    assert div.index == 1


def test_resync_found_after_a_transient_flip():
    a = make_record(["the", " cat", " sat", " on", " the", " mat"], name="a")
    b = make_record(["the", " dog", " sat", " on", " the", " mat"], name="b")
    div = first_divergence(a, b)
    resync = find_resync(a, b, div.index)
    assert resync is not None
    assert resync.a_index == 2
    assert resync.b_index == 2
    assert resync.length >= 3


def test_resync_absent_when_derailed_or_match_too_short():
    a = make_record(["x", "a1", "a2", "a3", "a4"], name="a")
    b = make_record(["x", "b1", "b2", "b3", "b4"], name="b")
    assert find_resync(a, b, 1) is None
    # A single shared token after the flip is coincidence, not realignment.
    a2 = make_record(["x", "a1", "same", "a2", "a3"], name="a2")
    b2 = make_record(["x", "b1", "same", "b2", "b3"], name="b2")
    assert find_resync(a2, b2, 1, min_len=3) is None


def test_prefix_drift_reports_mean_max_and_location():
    a = make_record(["x", "y", "z"], name="a", logprobs=[-1.0, -1.0, -1.0])
    b = make_record(["x", "y", "z"], name="b", logprobs=[-1.001, -1.05, -1.002])
    drift = prefix_drift(a, b, 3)
    assert drift.count == 3
    assert drift.max_index == 1
    assert drift.max_abs_delta == pytest.approx(0.05)
    assert drift.mean_abs_delta == pytest.approx((0.001 + 0.05 + 0.002) / 3)
    assert drift.trend == "flat"


def test_prefix_drift_handles_missing_logprobs():
    bare_a = make_record(["x", "y"], name="a")
    bare_b = make_record(["x", "y"], name="b")
    empty = prefix_drift(bare_a, bare_b, 2)
    assert empty.count == 0
    assert empty.trend == "none"
    # Positions missing a logprob on either side are skipped, not zeroed.
    partial_a = make_record(["x", "y"], name="pa", logprobs=[-1.0, None])
    partial_b = make_record(["x", "y"], name="pb", logprobs=[-1.2, -1.0])
    assert prefix_drift(partial_a, partial_b, 2).count == 1


def test_prefix_drift_detects_accumulation():
    # Error that grows along the sequence is the fingerprint of compounding
    # numerical noise; flat error is just independent jitter.
    n = 12
    lp_a = [-1.0] * n
    lp_b = [-1.0 - (0.001 if i < n // 2 else 0.02) for i in range(n)]
    texts = [f"t{i}" for i in range(n)]
    a = make_record(texts, name="a", logprobs=lp_a)
    b = make_record(texts, name="b", logprobs=lp_b)
    assert prefix_drift(a, b, n).trend == "accumulating"


def test_tie_analysis_flags_a_near_tie():
    tops = {1: [("y", -0.693), ("q", -0.694)]}
    a = make_record(["x", "y"], name="a", logprobs=[-0.1, -0.693], tops=tops)
    b = make_record(["x", "q"], name="b", logprobs=[-0.1, -0.694],
                    tops={1: [("q", -0.694), ("y", -0.695)]})
    div = first_divergence(a, b)
    tie = analyze_tie(a, b, div, epsilon=0.05)
    assert tie.available
    assert tie.near_tie
    assert tie.cross_gap == pytest.approx(0.001)


def test_tie_analysis_large_gap_is_not_a_tie():
    a = make_record(["x", "y"], name="a", logprobs=[-0.1, -0.1],
                    tops={1: [("y", -0.1), ("q", -3.0)]})
    b = make_record(["x", "q"], name="b", logprobs=[-0.1, -0.1],
                    tops={1: [("q", -0.1), ("y", -3.0)]})
    div = first_divergence(a, b)
    tie = analyze_tie(a, b, div, epsilon=0.05)
    assert tie.available
    assert not tie.near_tie
    assert tie.cross_gap == pytest.approx(2.9)


def test_tie_analysis_without_top_lists_or_with_absent_tokens():
    plain_a = make_record(["x", "y"], name="a")
    plain_b = make_record(["x", "q"], name="b")
    tie = analyze_tie(plain_a, plain_b, first_divergence(plain_a, plain_b))
    assert not tie.available
    # b's token never made a's top-k: the distribution moved a lot.
    top_a = make_record(["x", "y"], name="ta", logprobs=[-0.1, -0.1],
                        tops={1: [("y", -0.1), ("z", -1.0)]})
    tie = analyze_tie(top_a, plain_b, first_divergence(top_a, plain_b))
    assert tie.absent_from_top
