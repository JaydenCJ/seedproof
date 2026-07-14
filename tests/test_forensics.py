"""Forensic rule chain: every verdict, its confidence, and its evidence."""

from __future__ import annotations

import json

from conftest import make_record
from seedproof import VERDICTS, diagnose


def _labels(diagnosis):
    return [item.label for item in diagnosis.evidence]


def test_identical_runs_and_the_cross_config_reproducibility_win():
    same = diagnose(make_record(["x", "y"], name="a", backend="cpu"),
                    make_record(["x", "y"], name="b", backend="cpu"))
    assert same.verdict == "identical"
    assert same.confidence == "high"
    assert not same.divergence.diverged
    # Matching streams from *different* configs is the reproducibility
    # result users are hunting for; the report should say so explicitly.
    cross = diagnose(make_record(["x"], name="a", backend="cpu"),
                     make_record(["x"], name="b", backend="cuda"))
    assert cross.verdict == "identical"
    assert any("reproducibility win" in item.detail for item in cross.evidence)


def test_different_prompts_dominate_every_other_signal():
    # Even with config deltas present, a prompt mismatch must win: nothing
    # downstream is comparable when the runs answered different questions.
    a = make_record(["x"], name="a", prompt="p1", backend="cpu")
    b = make_record(["y"], name="b", prompt="p2", backend="cuda")
    dx = diagnose(a, b)
    assert dx.verdict == "prompt-mismatch"
    assert dx.confidence == "high"
    assert dx.evidence[0].label == "prompt"


def test_prompt_mismatch_wins_even_when_the_streams_happen_to_match():
    # Two runs of *different* prompts can coincidentally emit the same
    # tokens (short answers, refusals). Calling that "identical" would sell
    # a coincidence as a reproducibility result — the prompt rule must win
    # on the non-diverged path too.
    a = make_record(["Yes", "."], name="a", prompt="Is 2+2=4?", backend="cpu")
    b = make_record(["Yes", "."], name="b", prompt="Is water wet?", backend="cpu")
    dx = diagnose(a, b)
    assert dx.verdict == "prompt-mismatch"
    assert dx.confidence == "high"
    assert not dx.divergence.diverged
    assert dx.evidence[0].label == "prompt"
    assert "coincidence" in dx.summary


def test_render_diff_shows_the_real_verdict_for_non_diverged_mismatches():
    # The text report must print the diagnosis verdict, not a hardcoded
    # "identical", when the streams match but the runs are not comparable.
    from seedproof.report import render_diff

    a = make_record(["Yes"], name="a", prompt="q1")
    b = make_record(["Yes"], name="b", prompt="q2")
    lines = render_diff(diagnose(a, b), a, b)
    assert any("verdict: prompt-mismatch" in line for line in lines)
    assert not any("verdict: identical" in line for line in lines)


def test_tokenizer_boundary_covers_segmentation_and_id_renumbering():
    # Same decoded text, different segmentation:
    seg = diagnose(make_record(["hel", "lo"], name="a", with_ids=False),
                   make_record(["hell", "o"], name="b", with_ids=False))
    assert seg.verdict == "tokenizer-boundary"
    assert "not a model behavior difference" in seg.summary
    # Same texts position-by-position but renumbered ids:
    a = make_record(["x", "y"], name="a")
    b = make_record(["x", "y"], name="b")
    b.tokens[1].id = a.tokens[1].id + 7
    assert diagnose(a, b, basis="id").verdict == "tokenizer-boundary"


def test_sampler_settings_differences_win_over_runtime_axes():
    a = make_record(["x", "y"], name="a", backend="cpu", temperature=0.0)
    b = make_record(["x", "q"], name="b", backend="cuda",
                    sampler="top-p", temperature=0.8)
    dx = diagnose(a, b)
    assert dx.verdict == "sampler-config"
    assert "temperature" in dx.summary


def test_different_seeds_under_stochastic_sampling():
    a = make_record(["x", "y"], name="a", sampler="top-p", temperature=0.8, seed=1)
    b = make_record(["x", "q"], name="b", sampler="top-p", temperature=0.8, seed=2)
    dx = diagnose(a, b)
    assert dx.verdict == "seed-mismatch"
    assert dx.confidence == "high"
    assert "1 vs 2" in dx.summary


def test_greedy_divergence_is_nondeterminism_even_with_a_seed_delta():
    # Greedy decoding never consults the RNG: with no other delta the real
    # story is nondeterminism, and the seed evidence must say why.
    seeded = diagnose(make_record(["x", "y"], name="a", seed=1),
                      make_record(["x", "q"], name="b", seed=2))
    assert seeded.verdict == "nondeterminism"
    assert any("greedy" in item.detail
               for item in seeded.evidence if item.label == "seed")
    # And with a fully identical config the verdict is the same, high
    # confidence, pointing at the usual runtime suspects.
    twin = diagnose(make_record(["x", "y"], name="a", backend="cuda", seed=42),
                    make_record(["x", "q"], name="b", backend="cuda", seed=42))
    assert twin.verdict == "nondeterminism"
    assert twin.confidence == "high"
    assert any("atomic reductions" in item.detail for item in twin.evidence)


def test_different_models_is_its_own_verdict():
    a = make_record(["x", "y"], name="a", model="base-7b")
    b = make_record(["x", "q"], name="b", model="finetune-7b")
    dx = diagnose(a, b)
    assert dx.verdict == "model-mismatch"
    assert "base-7b" in dx.summary


def test_backend_confidence_depends_on_logprob_evidence():
    # With top-k logprobs the tie-break is provable -> high confidence.
    tops = {1: [("y", -0.69), ("q", -0.70)]}
    rich = diagnose(
        make_record(["x", "y"], name="a", backend="cpu",
                    logprobs=[-0.1, -0.69], tops=tops),
        make_record(["x", "q"], name="b", backend="cuda",
                    logprobs=[-0.1, -0.70], tops={1: [("q", -0.70), ("y", -0.71)]}),
    )
    assert rich.verdict == "backend-numerics"
    assert rich.confidence == "high"
    assert "tie-break" in _labels(rich)
    # Without logprobs the same delta caps at medium and says how to fix it.
    bare = diagnose(make_record(["x", "y"], name="a", backend="cpu"),
                    make_record(["x", "q"], name="b", backend="cuda"))
    assert bare.verdict == "backend-numerics"
    assert bare.confidence == "medium"
    assert any("re-capture with logprobs" in item.detail for item in bare.evidence)


def test_quant_only_delta_is_quant_numerics():
    a = make_record(["x", "y"], name="a", quant="fp16")
    b = make_record(["x", "q"], name="b", quant="q4_k_m")
    dx = diagnose(a, b)
    assert dx.verdict == "quant-numerics"
    assert "fp16 vs q4_k_m" in dx.summary


def test_runtime_config_for_multiple_axes_and_extra_keys():
    multi = diagnose(
        make_record(["x", "y"], name="a", backend="cpu", quant="fp16"),
        make_record(["x", "q"], name="b", backend="cuda", quant="q4_k_m"),
    )
    assert multi.verdict == "runtime-config"
    assert "backend" in multi.summary and "quant" in multi.summary
    # Free-form extra knobs (batch size, thread count) are runtime axes too.
    a = make_record(["x", "y"], name="a")
    b = make_record(["x", "q"], name="b")
    b.config.extra = {"batch_size": 32}
    extra = diagnose(a, b)
    assert extra.verdict == "runtime-config"
    assert "extra.batch_size" in extra.config_deltas


def test_stochastic_nondeterminism_is_only_medium_confidence():
    # With a stochastic sampler an unseeded runtime RNG is an equally good
    # explanation, so certainty must drop.
    a = make_record(["x", "y"], name="a", sampler="top-p", temperature=0.9, seed=7)
    b = make_record(["x", "q"], name="b", sampler="top-p", temperature=0.9, seed=7)
    dx = diagnose(a, b)
    assert dx.verdict == "nondeterminism"
    assert dx.confidence == "medium"
    assert any(item.label == "rng" for item in dx.evidence)


def test_gap_size_and_epsilon_control_the_tie_break_evidence():
    tops_a = {1: [("y", -0.5), ("q", -0.6)]}
    tops_b = {1: [("q", -0.6), ("y", -0.5)]}
    a = make_record(["x", "y"], name="a", quant="fp16",
                    logprobs=[-0.1, -0.5], tops=tops_a)
    b = make_record(["x", "q"], name="b", quant="q4_k_m",
                    logprobs=[-0.1, -0.6], tops=tops_b)
    loose = diagnose(a, b, tie_epsilon=0.5)
    assert loose.tie.near_tie
    assert any("tie-break" == item.label for item in loose.evidence)
    strict = diagnose(a, b, tie_epsilon=0.01)
    assert not strict.tie.near_tie
    assert any("distribution itself moved" in item.detail for item in strict.evidence)


def test_length_and_resync_evidence():
    trunc = diagnose(make_record(["x", "y", "z"], name="a", backend="cpu"),
                     make_record(["x", "y"], name="b", backend="cuda"))
    assert trunc.divergence.kind == "length"
    assert any("stop" in item.detail
               for item in trunc.evidence if item.label == "length")
    flip = diagnose(
        make_record(["the", " cat", " sat", " on", " the", " mat"],
                    name="a", backend="cpu"),
        make_record(["the", " dog", " sat", " on", " the", " mat"],
                    name="b", backend="cuda"),
    )
    assert any("reconverge for" in item.detail for item in flip.evidence)


def test_to_dict_is_json_serializable_and_verdicts_are_documented():
    tops = {1: [("y", -0.69), ("q", -0.70)]}
    a = make_record(["x", "y"], name="a", backend="cpu",
                    logprobs=[-0.1, -0.69], tops=tops)
    b = make_record(["x", "q"], name="b", backend="cuda",
                    logprobs=[-0.1, -0.70], tops={1: [("q", -0.70), ("y", -0.71)]})
    dx = diagnose(a, b)
    data = json.loads(json.dumps(dx.to_dict()))
    assert data["verdict"] == "backend-numerics"
    assert data["divergence"]["index"] == 1
    assert data["config_deltas"]["backend"] == {"a": "cpu", "b": "cuda"}
    assert data["tie"]["near_tie"] is True
    assert data["prefix_drift"]["count"] == 1
    # The VERDICTS tuple is the public contract for --json consumers.
    assert dx.verdict in VERDICTS
    assert "identical" in VERDICTS and "nondeterminism" in VERDICTS
