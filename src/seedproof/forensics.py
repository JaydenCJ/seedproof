"""Divergence forensics: turn a raw divergence into a cause, with evidence.

Given two run records and the alignment facts from :mod:`seedproof.align`,
:func:`diagnose` walks an ordered rule chain — cheap, certain explanations
first — and produces a :class:`Diagnosis`: a verdict code, a confidence
level, a one-line summary, and the itemized evidence that supports it.

Verdict codes (see the README's reference table):

- ``identical``          — no divergence; the streams match on the basis.
- ``prompt-mismatch``    — the runs answered different prompts.
- ``tokenizer-boundary`` — same generated text, segmented or numbered
                           differently; a tokenizer/vocab difference, not a
                           model behavior difference.
- ``seed-mismatch``      — stochastic sampling with different seeds.
- ``sampler-config``     — sampler settings differ (temperature/top-k/...).
- ``model-mismatch``     — different model identifiers.
- ``quant-numerics``     — same model, different quantization.
- ``backend-numerics``   — same weights, different backend/device.
- ``runtime-config``     — several runtime axes differ at once.
- ``nondeterminism``     — identical configs still diverged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .align import (
    DEFAULT_TIE_EPSILON,
    Divergence,
    PrefixDrift,
    Resync,
    TieAnalysis,
    analyze_tie,
    find_resync,
    first_divergence,
    prefix_drift,
)
from .record import RUNTIME_FIELDS, SAMPLING_FIELDS, RunRecord, compare_configs

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

#: Every verdict :func:`diagnose` can return, in rule order.
VERDICTS: Tuple[str, ...] = (
    "identical",
    "prompt-mismatch",
    "tokenizer-boundary",
    "seed-mismatch",
    "sampler-config",
    "model-mismatch",
    "quant-numerics",
    "backend-numerics",
    "runtime-config",
    "nondeterminism",
)


@dataclass
class Evidence:
    """One observed fact supporting (or qualifying) the verdict."""

    label: str
    detail: str


@dataclass
class Diagnosis:
    """The full forensic result for one pair of runs."""

    verdict: str
    confidence: str
    summary: str
    divergence: Divergence
    config_deltas: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    drift: Optional[PrefixDrift] = None
    tie: Optional[TieAnalysis] = None
    resync: Optional[Resync] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view (used by ``seedproof diff --json``)."""
        div = self.divergence
        data: Dict[str, Any] = {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "divergence": {
                "kind": div.kind,
                "basis": div.basis,
                "index": div.index,
                "common_len": div.common_len,
                "a_token": div.a_token.text if div.a_token else None,
                "b_token": div.b_token.text if div.b_token else None,
            },
            "config_deltas": {
                key: {"a": pair[0], "b": pair[1]}
                for key, pair in self.config_deltas.items()
            },
            "evidence": [{"label": e.label, "detail": e.detail} for e in self.evidence],
        }
        if self.drift is not None and self.drift.count:
            data["prefix_drift"] = {
                "count": self.drift.count,
                "max_abs_delta": self.drift.max_abs_delta,
                "mean_abs_delta": self.drift.mean_abs_delta,
                "max_index": self.drift.max_index,
                "trend": self.drift.trend,
            }
        if self.tie is not None and self.tie.available:
            data["tie"] = {
                "cross_gap": self.tie.cross_gap,
                "runner_up_gap": self.tie.runner_up_gap,
                "near_tie": self.tie.near_tie,
                "absent_from_top": self.tie.absent_from_top,
            }
        if self.resync is not None:
            data["resync"] = {
                "a_index": self.resync.a_index,
                "b_index": self.resync.b_index,
                "length": self.resync.length,
            }
        return data


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _count(count: int, noun: str) -> str:
    """``1 token`` / ``48 tokens`` — a count with a correctly pluralized noun."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _token_pair(divergence: Divergence) -> str:
    a = _quote(divergence.a_token.text) if divergence.a_token else "<end of stream>"
    b = _quote(divergence.b_token.text) if divergence.b_token else "<end of stream>"
    return f"{a} vs {b}"


def _where(divergence: Divergence) -> str:
    if divergence.kind == "length":
        return f"token {divergence.index} (one stream ends)"
    return f"token {divergence.index}"


def diagnose(
    a: RunRecord,
    b: RunRecord,
    basis: str = "auto",
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
) -> Diagnosis:
    """Compare two runs and explain the first divergence.

    The rule chain is ordered so that certain, structural explanations
    (different prompt, different sampler settings) win over numerical ones —
    there is no point blaming the GPU when the two runs asked different
    questions.
    """
    divergence = first_divergence(a, b, basis)
    deltas = compare_configs(a.config, b.config)

    if not divergence.diverged:
        # A prompt mismatch still wins here: two runs that answered different
        # prompts are not comparable, and a coincidental token match must not
        # be reported as a reproducibility result.
        if a.prompt_sha256() != b.prompt_sha256():
            return Diagnosis(
                verdict="prompt-mismatch",
                confidence=CONFIDENCE_HIGH,
                summary=(
                    "the two runs answered different prompts — the matching "
                    "token streams are a coincidence, not reproducibility"
                ),
                divergence=divergence,
                config_deltas=deltas,
                evidence=[
                    Evidence(
                        "prompt",
                        f"prompt sha256 {a.prompt_sha256()[:12]} vs "
                        f"{b.prompt_sha256()[:12]}",
                    )
                ],
            )
        diagnosis = Diagnosis(
            verdict="identical",
            confidence=CONFIDENCE_HIGH,
            summary=(
                f"runs are token-identical on the {divergence.basis} basis "
                f"({_count(len(a.tokens), 'token')})"
            ),
            divergence=divergence,
            config_deltas=deltas,
        )
        if deltas:
            diagnosis.evidence.append(
                Evidence(
                    "config",
                    "configs differ ("
                    + ", ".join(sorted(deltas))
                    + ") yet outputs match — a reproducibility win worth recording",
                )
            )
        return diagnosis

    drift = prefix_drift(a, b, divergence.common_len)
    tie = analyze_tie(a, b, divergence, epsilon=tie_epsilon)
    resync = None
    if divergence.kind == "token" and divergence.index is not None:
        resync = find_resync(a, b, divergence.index)

    diagnosis = Diagnosis(
        verdict="",
        confidence="",
        summary="",
        divergence=divergence,
        config_deltas=deltas,
        drift=drift,
        tie=tie,
        resync=resync,
    )
    _attach_shared_evidence(diagnosis, tie_epsilon)
    _classify(a, b, diagnosis)
    return diagnosis


# -- rule chain ---------------------------------------------------------------


def _classify(a: RunRecord, b: RunRecord, dx: Diagnosis) -> None:
    div, deltas = dx.divergence, dx.config_deltas
    where, pair = _where(div), _token_pair(div)

    # Rule 1: different prompts — nothing downstream is comparable.
    if a.prompt_sha256() != b.prompt_sha256():
        dx.verdict, dx.confidence = "prompt-mismatch", CONFIDENCE_HIGH
        dx.summary = (
            "the two runs answered different prompts — "
            "any token divergence is expected and uninformative"
        )
        dx.evidence.insert(
            0,
            Evidence(
                "prompt",
                f"prompt sha256 {a.prompt_sha256()[:12]} vs {b.prompt_sha256()[:12]}",
            ),
        )
        return

    # Rule 2: token streams differ but the decoded text is identical.
    if a.decoded() == b.decoded():
        dx.verdict, dx.confidence = "tokenizer-boundary", CONFIDENCE_HIGH
        dx.summary = (
            f"identical generated text, different token streams from {where} — "
            "a tokenizer/vocab difference, not a model behavior difference"
        )
        dx.evidence.insert(
            0, Evidence("decoded", "decoded outputs are byte-identical")
        )
        return

    stochastic = a.config.is_stochastic() and b.config.is_stochastic()

    # Rule 3: sampler settings differ — different token-picking rules.
    sampling_deltas = [f for f in SAMPLING_FIELDS if f in deltas]
    if sampling_deltas:
        dx.verdict, dx.confidence = "sampler-config", CONFIDENCE_HIGH
        dx.summary = (
            f"sampler settings differ ({', '.join(sampling_deltas)}); "
            f"first divergence at {where}: {pair}"
        )
        return

    # Rule 4: different seeds under stochastic sampling.
    if "seed" in deltas:
        if stochastic:
            dx.verdict, dx.confidence = "seed-mismatch", CONFIDENCE_HIGH
            dx.summary = (
                f"stochastic sampling with different seeds "
                f"({deltas['seed'][0]} vs {deltas['seed'][1]}); "
                f"first divergence at {where}: {pair}"
            )
            return
        dx.evidence.append(
            Evidence(
                "seed",
                f"seeds differ ({deltas['seed'][0]} vs {deltas['seed'][1]}) but the "
                "sampler is greedy — the seed cannot explain a greedy divergence",
            )
        )

    # Rule 5: different model identifiers.
    if "model" in deltas:
        dx.verdict, dx.confidence = "model-mismatch", CONFIDENCE_HIGH
        dx.summary = (
            f"different models ({_quote(deltas['model'][0])} vs "
            f"{_quote(deltas['model'][1])}); first divergence at {where}: {pair}"
        )
        return

    # Rule 6: runtime axes — where the numerics ran.
    runtime_deltas = [f for f in deltas if f in RUNTIME_FIELDS or f.startswith("extra.")]
    if runtime_deltas:
        has_lp = dx.tie is not None and dx.tie.available
        dx.confidence = CONFIDENCE_HIGH if has_lp else CONFIDENCE_MEDIUM
        if not has_lp:
            dx.evidence.append(
                Evidence(
                    "logprobs",
                    "no top-k logprobs in the records — re-capture with logprobs "
                    "to confirm the tie-break; confidence capped at medium",
                )
            )
        if runtime_deltas == ["quant"]:
            dx.verdict = "quant-numerics"
            dx.summary = (
                f"same weights at different precision "
                f"({deltas['quant'][0]} vs {deltas['quant'][1]}) "
                f"first disagree at {where}: {pair}"
            )
        elif all(f in ("backend", "device") for f in runtime_deltas):
            axis = " and ".join(runtime_deltas)
            dx.verdict = "backend-numerics"
            dx.summary = (
                f"same config on a different {axis}; "
                f"first divergence at {where}: {pair}"
            )
        else:
            dx.verdict = "runtime-config"
            dx.summary = (
                f"runtime configuration differs ({', '.join(runtime_deltas)}); "
                f"first divergence at {where}: {pair}"
            )
        if stochastic:
            dx.evidence.append(
                Evidence(
                    "rng",
                    "sampling is stochastic: identical seeds only reproduce when "
                    "both runtimes implement the same RNG stream",
                )
            )
        return

    # Rule 7: nothing differs on paper, yet the streams diverged.
    dx.verdict = "nondeterminism"
    dx.confidence = CONFIDENCE_HIGH if not stochastic else CONFIDENCE_MEDIUM
    dx.summary = (
        f"identical configuration (fingerprint {a.config.fingerprint()}) still "
        f"diverged at {where}: {pair} — the runtime itself is nondeterministic"
    )
    dx.evidence.append(
        Evidence(
            "runtime",
            "look for nondeterministic kernels: atomic reductions, "
            "batch-size-dependent kernel selection, or thread scheduling",
        )
    )
    if stochastic:
        dx.evidence.append(
            Evidence(
                "rng",
                "stochastic sampler: an unseeded or time-seeded RNG in the "
                "runtime would also produce this signature",
            )
        )


# -- shared numerical evidence ------------------------------------------------


def _attach_shared_evidence(dx: Diagnosis, epsilon: float) -> None:
    """Evidence that is factual regardless of which rule fires."""
    for key, (va, vb) in dx.config_deltas.items():
        dx.evidence.append(
            Evidence("config", f"{key}: {json.dumps(va)} -> {json.dumps(vb)}")
        )
    tie = dx.tie
    if tie is not None and tie.available and tie.cross_gap is not None:
        if tie.near_tie:
            dx.evidence.append(
                Evidence(
                    "tie-break",
                    f"the losing token trailed the winner by only "
                    f"{abs(tie.cross_gap):.4f} nats (<= epsilon {epsilon:g}) — "
                    "a numerical tie-break",
                )
            )
        else:
            dx.evidence.append(
                Evidence(
                    "gap",
                    f"the losing token trailed by {abs(tie.cross_gap):.4f} nats "
                    f"(> epsilon {epsilon:g}) — the distribution itself moved, "
                    "not just a tie-break",
                )
            )
    elif tie is not None and tie.absent_from_top:
        dx.evidence.append(
            Evidence(
                "top-k",
                "the other run's token does not appear in this run's top-k at the "
                "divergence point — a large distribution shift",
            )
        )
    drift = dx.drift
    if drift is not None and drift.count:
        detail = (
            f"mean |dlogprob| {drift.mean_abs_delta:.4f} over "
            f"{_count(drift.count, 'shared token')}, "
            f"max {drift.max_abs_delta:.4f} at token {drift.max_index}"
        )
        if drift.trend == "accumulating":
            detail += " — drift accumulates along the prefix"
        dx.evidence.append(Evidence("prefix-drift", detail))
    if dx.divergence.kind == "token":
        if dx.resync is not None:
            dx.evidence.append(
                Evidence(
                    "resync",
                    f"streams reconverge for {dx.resync.length}+ tokens at "
                    f"a[{dx.resync.a_index}] / b[{dx.resync.b_index}] — "
                    "a transient flip, not a full derail",
                )
            )
        else:
            dx.evidence.append(
                Evidence("resync", "streams never reconverge after the divergence")
            )
    elif dx.divergence.kind == "length":
        dx.evidence.append(
            Evidence(
                "length",
                "one stream is a strict prefix of the other — check stop "
                "conditions and max-token limits before blaming numerics",
            )
        )
