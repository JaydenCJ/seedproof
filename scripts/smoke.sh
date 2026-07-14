#!/usr/bin/env bash
# Smoke test for seedproof: generate the deterministic demo matrix, then
# exercise every CLI subcommand end-to-end — ingest, show, ls, diff (text +
# json), matrix, check — asserting on real output and exit codes.
# Self-contained: pure stdlib, no network, idempotent (works from a clean tree).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

# The package has zero runtime dependencies, so running from src/ needs no install.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/seedproof-smoke.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

echo "[smoke] python: $("$PYTHON" --version 2>&1)"

# 1. Generate the deterministic demo matrix (plus the nondeterminism pair).
"$PYTHON" "$ROOT/examples/make_runs.py" "$WORKDIR/runs" --racy >/dev/null \
  || fail "make_runs.py exited non-zero"
[ -f "$WORKDIR/runs/cpu-fp32-seed42.json" ] || fail "baseline record missing"

# 2. diff: identical runs exit 0 and say so.
ident_out="$("$PYTHON" -m seedproof diff \
  "$WORKDIR/runs/cpu-fp32-seed42.json" "$WORKDIR/runs/cpu-fp32-rerun.json")" \
  || fail "diff of identical runs should exit 0"
echo "$ident_out" | grep -q "verdict: identical" || fail "identical verdict missing"

# 3. diff: greedy runs with different seeds are still identical (seed ignored).
"$PYTHON" -m seedproof diff \
  "$WORKDIR/runs/cpu-fp32-seed42.json" "$WORKDIR/runs/cpu-fp32-seed7.json" >/dev/null \
  || fail "greedy runs with different seeds should be identical (exit 0)"

# 4. diff: cpu vs cuda pinpoints the flip and blames the backend, exit 1.
set +e
diff_out="$("$PYTHON" -m seedproof diff \
  "$WORKDIR/runs/cpu-fp32-seed42.json" "$WORKDIR/runs/cuda-fp32-seed42.json")"
diff_rc=$?
set -e
echo "$diff_out" | sed 's/^/[diff] /'
[ "$diff_rc" -eq 1 ] || fail "diff on divergence should exit 1, got $diff_rc"
echo "$diff_out" | grep -q "first divergent token: index 16" \
  || fail "divergence index not pinpointed"
echo "$diff_out" | grep -q "verdict: backend-numerics (confidence: high)" \
  || fail "backend-numerics verdict missing"
echo "$diff_out" | grep -q "tie-break" || fail "tie-break evidence missing"

# 5. diff: cpu vs q4 blames quantization.
set +e
quant_out="$("$PYTHON" -m seedproof diff \
  "$WORKDIR/runs/cpu-fp32-seed42.json" "$WORKDIR/runs/cpu-q4-seed42.json")"
set -e
echo "$quant_out" | grep -q "verdict: quant-numerics" || fail "quant-numerics verdict missing"

# 6. diff: identical configs that still diverge -> nondeterminism.
set +e
racy_out="$("$PYTHON" -m seedproof diff \
  "$WORKDIR/runs/racy-a.json" "$WORKDIR/runs/racy-b.json")"
set -e
echo "$racy_out" | grep -q "verdict: nondeterminism" || fail "nondeterminism verdict missing"

# 7. diff --json: machine output parses and carries the verdict.
"$PYTHON" -m seedproof diff --json \
  "$WORKDIR/runs/cpu-fp32-seed42.json" "$WORKDIR/runs/cuda-fp32-seed42.json" \
  > "$WORKDIR/diff.json" || true
"$PYTHON" - "$WORKDIR/diff.json" <<'EOF' || fail "diff --json is not valid or wrong"
import json, sys
data = json.load(open(sys.argv[1]))
assert data["verdict"] == "backend-numerics", data["verdict"]
assert data["divergence"]["index"] == 16, data["divergence"]
assert data["tie"]["near_tie"] is True, data["tie"]
EOF

# 8. matrix: 5 runs, 3 classes, combined axes explain the split.
matrix_out="$("$PYTHON" -m seedproof matrix \
  "$WORKDIR/runs/cpu-fp32-seed42.json" "$WORKDIR/runs/cpu-fp32-rerun.json" \
  "$WORKDIR/runs/cpu-fp32-seed7.json" "$WORKDIR/runs/cuda-fp32-seed42.json" \
  "$WORKDIR/runs/cpu-q4-seed42.json")" || fail "matrix exited non-zero"
echo "$matrix_out" | sed 's/^/[matrix] /'
echo "$matrix_out" | grep -q "classes: 3" || fail "matrix did not find 3 classes"
echo "$matrix_out" | grep -q "backend + quant together explain the split" \
  || fail "combined-axis analysis missing"

# 9. check: the CI gate passes on reproducible runs and fails on the full dir.
"$PYTHON" -m seedproof check \
  "$WORKDIR/runs/cpu-fp32-seed42.json" "$WORKDIR/runs/cpu-fp32-rerun.json" \
  "$WORKDIR/runs/cpu-fp32-seed7.json" | grep -q "^OK:" \
  || fail "check should pass on reproducible runs"
set +e
"$PYTHON" -m seedproof check "$WORKDIR/runs" >/dev/null
check_rc=$?
set -e
[ "$check_rc" -eq 1 ] || fail "check on the full dir should exit 1, got $check_rc"

# 10. ingest: a captured SSE stream becomes a canonical record.
"$PYTHON" -m seedproof ingest "$ROOT/examples/sse-capture.txt" --format sse \
  --backend cuda --quant q4_k_m --seed 42 --prompt "Why is the sky blue?" \
  -o "$WORKDIR/run-gpu.json" >/dev/null || fail "sse ingest failed"
show_out="$("$PYTHON" -m seedproof show "$WORKDIR/run-gpu.json")"
echo "$show_out" | grep -q "tokens: 5  (with logprobs)" || fail "ingested record wrong"
echo "$show_out" | grep -q "model=local-8b-q4" || fail "model not extracted from capture"

# 11. ls: the record table lists every run with a stream fingerprint.
ls_out="$("$PYTHON" -m seedproof ls "$WORKDIR/runs")"
echo "$ls_out" | grep -E 'cpu-q4-seed42\.json +cpu +q4_k_m +42 +48 ' >/dev/null \
  || fail "ls did not report the q4 run"

# 12. --version and --help agree with the package.
version_out="$("$PYTHON" -m seedproof --version)"
pkg_version="$("$PYTHON" -c 'import seedproof; print(seedproof.__version__)')"
[ "$version_out" = "seedproof $pkg_version" ] \
  || fail "--version mismatch: '$version_out' vs package '$pkg_version'"
"$PYTHON" -m seedproof --help | grep -q "matrix" || fail "--help missing matrix command"

echo "SMOKE OK"
