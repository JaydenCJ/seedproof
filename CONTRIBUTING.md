# Contributing to seedproof

Thanks for your interest in contributing. Issues, discussions, and pull
requests are all welcome.

## Getting started

You need Python ≥ 3.9; there are no other runtime prerequisites.

```bash
git clone https://github.com/JaydenCJ/seedproof
cd seedproof
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks

```bash
pytest                 # unit + CLI tests, all offline and deterministic
bash scripts/smoke.sh  # end-to-end: generate a matrix, ingest, diff, check
```

Both must pass before a pull request is reviewed; the smoke script exercises
the real CLI against the generated demo matrix and must print `SMOKE OK`.

## Before you open a pull request

1. Format touched files consistently with the surrounding code (PEP 8,
   4-space indents, type hints on public functions).
2. `pytest` must pass with zero failures — the suite is fast (< 1 s).
3. `bash scripts/smoke.sh` must print `SMOKE OK`.
4. Add tests for behavior changes; keep logic in the pure modules
   (`record.py`, `align.py`, `forensics.py`, `matrix.py`) so it stays
   unit-testable without touching the CLI.

## Ground rules

- **No new runtime dependencies.** The package is standard-library only;
  that is a feature. Test-only dependencies belong in the `dev` extra.
- **Record format changes need a version bump and docs.** Anything that
  changes the meaning of an existing field must bump `RECORD_VERSION` and
  update `docs/record-format.md` in the same pull request.
- **Verdicts must never overstate certainty.** A new forensic rule needs a
  test for the evidence it emits and for the confidence level it maps to;
  when the logprob evidence is missing, confidence is capped at `medium`.
- No network calls, no telemetry; everything runs offline and deterministic.
- Code comments and doc comments are written in English.

## Reporting bugs

Please include `seedproof --version` output, the exact command, and the two
record files (records contain only the prompt, config, and token stream —
no weights). If a verdict seems wrong, say which cause you expected and
which evidence line convinced you.

## Security

Please do not report security issues in public GitHub issues. Use GitHub's
private vulnerability reporting on this repository instead.
