"""The shipped examples are executable documentation: pin their behavior.

The simulated runtime in ``examples/make_runs.py`` derives every logit from
SHA-256 hashes, so its output is bit-reproducible across machines — these
tests pin the exact divergence indices the README and smoke test rely on.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from seedproof import build_matrix, diagnose, load

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


@pytest.fixture(scope="module")
def make_runs():
    spec = importlib.util.spec_from_file_location(
        "make_runs", os.path.join(EXAMPLES, "make_runs.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_runs"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_dir(make_runs, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("demo-runs")
    make_runs.main([str(outdir), "--racy"])
    return outdir


def test_generator_is_deterministic_across_invocations(make_runs, tmp_path):
    make_runs.main([str(tmp_path / "one")])
    make_runs.main([str(tmp_path / "two")])
    for name in os.listdir(tmp_path / "one"):
        with open(tmp_path / "one" / name, encoding="utf-8") as f_one:
            with open(tmp_path / "two" / name, encoding="utf-8") as f_two:
                assert f_one.read() == f_two.read(), name


def test_reruns_and_greedy_seed_changes_are_identical(demo_dir):
    baseline = load(str(demo_dir / "cpu-fp32-seed42.json"))
    rerun = load(str(demo_dir / "cpu-fp32-rerun.json"))
    assert diagnose(baseline, rerun).verdict == "identical"
    # Greedy decoding never consults the RNG, so the seed must not matter.
    other_seed = load(str(demo_dir / "cpu-fp32-seed7.json"))
    assert diagnose(baseline, other_seed).verdict == "identical"


def test_backend_divergence_is_pinned_at_token_16(demo_dir):
    # This exact index appears in the README quickstart and the smoke test;
    # if the simulation constants change, all three must move together.
    a = load(str(demo_dir / "cpu-fp32-seed42.json"))
    b = load(str(demo_dir / "cuda-fp32-seed42.json"))
    dx = diagnose(a, b)
    assert dx.verdict == "backend-numerics"
    assert dx.confidence == "high"
    assert dx.divergence.index == 16
    assert dx.tie.near_tie


def test_quant_divergence_is_pinned_at_token_17(demo_dir):
    a = load(str(demo_dir / "cpu-fp32-seed42.json"))
    b = load(str(demo_dir / "cpu-q4-seed42.json"))
    dx = diagnose(a, b)
    assert dx.verdict == "quant-numerics"
    assert dx.divergence.index == 17


def test_racy_pair_diagnoses_nondeterminism(demo_dir):
    a = load(str(demo_dir / "racy-a.json"))
    b = load(str(demo_dir / "racy-b.json"))
    dx = diagnose(a, b)
    assert dx.verdict == "nondeterminism"
    assert not dx.config_deltas


def test_demo_matrix_splits_three_ways_explained_by_two_axes(demo_dir):
    names = ["cpu-fp32-seed42", "cpu-fp32-rerun", "cpu-fp32-seed7",
             "cuda-fp32-seed42", "cpu-q4-seed42"]
    records = [load(str(demo_dir / f"{n}.json")) for n in names]
    report = build_matrix(records)
    assert len(report.classes) == 3
    assert report.classes[0].members == names[:3]
    assert report.combined_axes == ("backend", "quant")
