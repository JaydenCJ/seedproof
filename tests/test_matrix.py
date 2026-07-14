"""Run matrix: equivalence classes, axis analysis, pairwise divergence."""

from __future__ import annotations

import pytest

from conftest import make_record
from seedproof import MatrixError, build_matrix
from seedproof.matrix import _class_labels


def test_identical_runs_form_one_class():
    runs = [make_record(["x", "y"], name=f"r{i}") for i in range(3)]
    report = build_matrix(runs)
    assert report.consistent
    assert len(report.classes) == 1
    assert report.classes[0].members == ["r0", "r1", "r2"]
    assert report.pairwise == []


def test_classes_are_labelled_by_size_descending():
    runs = [
        make_record(["x", "q"], name="lone", backend="cuda"),
        make_record(["x", "y"], name="m1", backend="cpu"),
        make_record(["x", "y"], name="m2", backend="cpu"),
    ]
    report = build_matrix(runs)
    assert [cls.label for cls in report.classes] == ["A", "B"]
    assert report.classes[0].members == ["m1", "m2"]  # majority first
    assert report.classes[1].members == ["lone"]


def test_input_errors_are_hard_and_specific():
    with pytest.raises(MatrixError, match="at least 2"):
        build_matrix([make_record(["x"], name="only")])
    # Comparing generations of different prompts is meaningless; the error
    # names the offending runs so the user can fix the capture set.
    mixed = [
        make_record(["x"], name="p1-run", prompt="p1"),
        make_record(["x"], name="p2-run", prompt="p2"),
    ]
    with pytest.raises(MatrixError, match="p2-run"):
        build_matrix(mixed)
    same = [make_record(["x"], name="a"), make_record(["x"], name="b")]
    with pytest.raises(MatrixError, match="unknown basis"):
        build_matrix(same, basis="bytes")


def test_axis_that_perfectly_explains_the_split():
    runs = [
        make_record(["x", "y"], name="c1", model="m", backend="cpu"),
        make_record(["x", "y"], name="c2", model="m", backend="cpu"),
        make_record(["x", "q"], name="g1", model="m", backend="cuda"),
    ]
    report = build_matrix(runs)
    # Constant fields (model) are not axes; only backend varies.
    assert [a.field for a in report.axes] == ["backend"]
    backend_axis = report.axes[0]
    assert backend_axis.relation == "explains"
    assert backend_axis.value_classes == {'"cpu"': ["A"], '"cuda"': ["B"]}


def test_axis_with_a_value_in_two_classes_is_mixed():
    # backend=cpu produced both streams, so backend alone cannot explain it.
    runs = [
        make_record(["x", "y"], name="a", backend="cpu", quant="fp32"),
        make_record(["x", "q"], name="b", backend="cpu", quant="q4"),
        make_record(["x", "y"], name="c", backend="cuda", quant="fp32"),
    ]
    report = build_matrix(runs)
    backend_axis = next(a for a in report.axes if a.field == "backend")
    assert backend_axis.relation == "mixed"


def test_axis_that_correlates_but_is_coarser_than_the_split():
    # Three seed values map cleanly onto two classes: same value never spans
    # classes (a function), but the partitions are not equal.
    runs = [
        make_record(["x", "y"], name="s1", sampler="top-p", temperature=0.8, seed=1),
        make_record(["x", "y"], name="s2", sampler="top-p", temperature=0.8, seed=2),
        make_record(["x", "q"], name="s3", sampler="top-p", temperature=0.8, seed=3),
    ]
    report = build_matrix(runs)
    seed_axis = next(a for a in report.axes if a.field == "seed")
    assert seed_axis.relation == "correlates"


def test_combined_axes_only_when_no_single_field_explains():
    split = [
        make_record(["x", "y"], name="cpu32", backend="cpu", quant="fp32"),
        make_record(["x", "q"], name="gpu32", backend="cuda", quant="fp32"),
        make_record(["x", "z"], name="cpu4", backend="cpu", quant="q4"),
    ]
    report = build_matrix(split)
    assert all(a.relation != "explains" for a in report.axes)
    assert report.combined_axes == ("backend", "quant")
    # When one field already explains everything, no pair is reported.
    simple = [
        make_record(["x", "y"], name="c1", backend="cpu"),
        make_record(["x", "q"], name="g1", backend="cuda"),
    ]
    assert build_matrix(simple).combined_axes is None


def test_pairwise_reports_first_divergence_between_classes():
    runs = [
        make_record(["x", "y", "z"], name="a", backend="cpu"),
        make_record(["x", "q", "z"], name="b", backend="cuda"),
    ]
    report = build_matrix(runs)
    pair = report.pairwise[0]
    assert (pair.label_a, pair.label_b) == ("A", "B")
    assert pair.index == 1
    assert (pair.a_text, pair.b_text) == ("y", "q")


def test_auto_basis_drops_to_text_when_any_record_lacks_ids():
    runs = [
        make_record(["x"], name="ids"),
        make_record(["x"], name="noids", with_ids=False),
    ]
    report = build_matrix(runs)
    assert report.basis == "text"
    assert report.consistent


def test_class_labels_extend_past_z():
    labels = _class_labels(28)
    assert labels[0] == "A"
    assert labels[25] == "Z"
    assert labels[26] == "AA"
    assert labels[27] == "AB"
    assert len(set(labels)) == 28
