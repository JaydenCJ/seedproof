#!/usr/bin/env python3
"""Generate a deterministic demo matrix of seedproof run records.

This simulates a tiny word-level "model" whose logits are derived from
SHA-256 hashes of the context, so every run is exactly reproducible with
the standard library alone — no weights, no network. Runtime effects are
emulated the way they show up in real life:

- ``cuda`` backend: a tiny deterministic per-position perturbation on every
  logit (what reordered floating-point reductions do to a kernel);
- ``q4_k_m`` quant: logits snapped to a coarse grid (what low-precision
  weights do to the distribution);
- ``--racy``: a hidden per-run nonce perturbs the logits *without* appearing
  in the config — the signature of true runtime nondeterminism.

Because generation is autoregressive over the simulated context, one flipped
token derails everything after it — exactly like a real model.

Usage: python3 examples/make_runs.py OUTDIR [--tokens N] [--racy]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from typing import List, Optional, Tuple

try:
    from seedproof import RunConfig, RunRecord, Token, TokenChoice, save
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from seedproof import RunConfig, RunRecord, Token, TokenChoice, save

PROMPT = "Explain why the sky is blue in one short paragraph."
MODEL = "sim-tinylab-1"

#: Word-level vocabulary of the simulated model (token id = list index).
VOCAB: List[str] = [
    " the", " a", " an", " of", " in", " and", " to", " is", " it", " that",
    " light", " sky", " blue", " sun", " air", " waves", " color", " eye",
    " short", " long", " scatter", " scatters", " because", " more", " most",
    " when", " through", " small", " molecules", " particles", " reaches",
    " appears", " looks", " white", " red", " sunset", " horizon", " energy",
    " wavelength", " wavelengths", " than", " so", " we", " see", " this",
    ",", ".", " which",
]

#: Magnitude of the emulated backend / nondeterminism perturbation (logits).
PERTURB = 0.02
#: Grid step of the emulated 4-bit quantization (logits).
QGRID = 0.06
#: Spread of the base logits.
SCALE = 6.0
#: Candidates recorded per token.
TOP_K = 4


def _unit(key: str) -> float:
    """Deterministic uniform in [0, 1) derived from a SHA-256 hash."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def _logits(context: List[str], backend: str, quant: str, nonce: int) -> List[float]:
    ctx = "|".join(context)
    position = len(context)
    logits = []
    for token_text in VOCAB:
        value = _unit(f"L|{PROMPT}|{ctx}|{token_text}") * SCALE
        if backend == "cuda":
            value += (_unit(f"P|cuda|{position}|{token_text}") - 0.5) * PERTURB
        if nonce:
            value += (_unit(f"N|{nonce}|{position}|{token_text}") - 0.5) * PERTURB
        if quant == "q4_k_m":
            value = round(value / QGRID) * QGRID
        logits.append(value)
    return logits


def _log_softmax(logits: List[float]) -> List[float]:
    peak = max(logits)
    log_z = peak + math.log(sum(math.exp(v - peak) for v in logits))
    return [v - log_z for v in logits]


def _pick(logprobs: List[float], sampler: str, temperature: float, u: float) -> int:
    if sampler == "greedy" or temperature <= 0.0:
        return max(range(len(logprobs)), key=logprobs.__getitem__)
    weights = [math.exp(lp / temperature) for lp in logprobs]
    total = sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight / total
        if u < cumulative:
            return index
    return len(logprobs) - 1


def simulate(
    backend: str,
    quant: str,
    seed: Optional[int],
    n_tokens: int,
    sampler: str = "greedy",
    temperature: float = 0.0,
    nonce: int = 0,
) -> List[Token]:
    """Run the simulated model and return its token stream with logprobs."""
    context: List[str] = []
    tokens: List[Token] = []
    for step in range(n_tokens):
        logprobs = _log_softmax(_logits(context, backend, quant, nonce))
        # A seeded, reproducible uniform per step for stochastic sampling.
        u = _unit(f"S|{seed}|{step}")
        chosen = _pick(logprobs, sampler, temperature, u)
        ranked = sorted(range(len(VOCAB)), key=logprobs.__getitem__, reverse=True)
        top = [
            TokenChoice(text=VOCAB[i], logprob=round(logprobs[i], 6), id=i)
            for i in ranked[:TOP_K]
        ]
        tokens.append(
            Token(
                text=VOCAB[chosen],
                id=chosen,
                logprob=round(logprobs[chosen], 6),
                top=top,
            )
        )
        context.append(VOCAB[chosen])
    return tokens


def make_record(
    name: str,
    backend: str,
    quant: str,
    seed: int,
    n_tokens: int,
    sampler: str = "greedy",
    temperature: float = 0.0,
    nonce: int = 0,
    notes: str = "",
) -> RunRecord:
    config = RunConfig(
        model=MODEL,
        backend=backend,
        quant=quant,
        seed=seed,
        sampler=sampler,
        temperature=temperature,
    )
    tokens = simulate(backend, quant, seed, n_tokens, sampler, temperature, nonce)
    return RunRecord(prompt=PROMPT, config=config, tokens=tokens, name=name, notes=notes)


#: The demo matrix: (filename stem, backend, quant, seed, nonce).
MATRIX: List[Tuple[str, str, str, int, int]] = [
    ("cpu-fp32-seed42", "cpu", "fp32", 42, 0),
    ("cpu-fp32-rerun", "cpu", "fp32", 42, 0),
    ("cpu-fp32-seed7", "cpu", "fp32", 7, 0),
    ("cuda-fp32-seed42", "cuda", "fp32", 42, 0),
    ("cpu-q4-seed42", "cpu", "q4_k_m", 42, 0),
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", help="directory to write the record files into")
    parser.add_argument("--tokens", type=int, default=48, help="tokens per run (default: 48)")
    parser.add_argument(
        "--racy",
        action="store_true",
        help="also write racy-a/racy-b: identical configs, hidden nondeterminism",
    )
    args = parser.parse_args(argv)

    written = []
    for name, backend, quant, seed, nonce in MATRIX:
        record = make_record(name, backend, quant, seed, args.tokens, nonce=nonce)
        path = os.path.join(args.outdir, f"{name}.json")
        save(record, path)
        written.append(path)
    if args.racy:
        for name, nonce in (("racy-a", 1), ("racy-b", 2)):
            record = make_record(
                name, "cuda", "fp32", 42, args.tokens, nonce=nonce,
                notes="simulated nondeterministic kernel (hidden per-run nonce)",
            )
            path = os.path.join(args.outdir, f"{name}.json")
            save(record, path)
            written.append(path)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
