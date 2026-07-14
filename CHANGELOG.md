# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-12

### Added

- Versioned JSON run-record format (`seedproof_record: 1`) capturing prompt,
  runtime config (model, backend, device, quant, seed, sampler settings,
  free-form `extra`), and the token stream with optional ids, logprobs, and
  top-k candidates. Sorted keys, atomic writes, tamper-detecting prompt
  hash, documented in `docs/record-format.md`.
- Ingest adapters for three capture formats: `generic` JSON (with key
  synonyms), `jsonl` token events, and `sse` — captured OpenAI-compatible
  streaming responses, including per-token `logprobs`/`top_logprobs`.
- Alignment engine: first divergent token on an id or text basis (`auto`
  prefers ids), strict-prefix detection, reconvergence search after the
  flip, shared-prefix logprob drift metrics, and near-tie analysis at the
  divergence point.
- Forensic rule chain producing one of ten verdicts (`identical`,
  `prompt-mismatch`, `tokenizer-boundary`, `seed-mismatch`,
  `sampler-config`, `model-mismatch`, `quant-numerics`, `backend-numerics`,
  `runtime-config`, `nondeterminism`) with a confidence level and itemized
  evidence; confidence is capped at `medium` when logprob evidence is
  missing.
- Run matrix: equivalence classes by output stream, per-axis analysis of
  which config field explains the split (including combined two-axis
  explanations), and pairwise first-divergence between classes.
- `seedproof` CLI: `ingest`, `show`, `ls`, `diff` (context window, `--json`,
  `--tie-epsilon`, `--basis`), `matrix` (`--json`), and `check` — a CI gate
  with `diff(1)`-style exit codes (0 identical, 1 divergent, 2 error).
- Deterministic divergence lab in `examples/make_runs.py` (simulated
  backend perturbation, quant grid rounding, hidden-nonce nondeterminism)
  and a real SSE capture sample in `examples/sse-capture.txt`.
- 90 offline pytest tests and `scripts/smoke.sh`, an end-to-end CLI smoke
  run that must print `SMOKE OK`.

### Notes

- The repository ships no CI workflow; verification is local —
  `pip install -e '.[dev]' && pytest && bash scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/seedproof/releases/tag/v0.1.0
