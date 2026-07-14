# seedproof examples

Everything here runs offline with the standard library — no model, no
server, no network.

## `make_runs.py` — a reproducible divergence lab

Generates a five-run demo matrix from a simulated model whose logits are
SHA-256-derived, so the records are byte-for-byte reproducible on any
machine. The simulation emulates the real failure modes seedproof exists to
diagnose:

| Run | Emulates |
|---|---|
| `cpu-fp32-seed42` | the baseline capture |
| `cpu-fp32-rerun` | a second run of the same config (identical) |
| `cpu-fp32-seed7` | a different seed under greedy decoding (identical — greedy ignores the seed) |
| `cuda-fp32-seed42` | GPU kernels: tiny logit perturbations that flip a near-tie |
| `cpu-q4-seed42` | 4-bit quantization: logits snapped to a coarse grid |

```bash
python3 examples/make_runs.py demo-runs
seedproof matrix demo-runs
seedproof diff demo-runs/cpu-fp32-seed42.json demo-runs/cuda-fp32-seed42.json
```

Add `--racy` to also write `racy-a`/`racy-b`: two runs with *identical*
configs whose logits are perturbed by a hidden per-run nonce — the signature
of a nondeterministic kernel, and the one case `seedproof diff` labels
`nondeterminism`.

## `sse-capture.txt` — ingest a real streaming capture

A captured OpenAI-compatible `chat/completions` stream (with
`logprobs` + `top_logprobs`), as produced by
`curl -sN http://127.0.0.1:8080/v1/chat/completions ... > sse-capture.txt`
against a local inference server. Convert it into a canonical record:

```bash
seedproof ingest examples/sse-capture.txt --format sse \
  --backend cuda --quant q4_k_m --seed 42 \
  --prompt "Why is the sky blue?" -o run-gpu.json
seedproof show run-gpu.json
```

Capture the same request from a second configuration (CPU build, different
quant, another machine), ingest it the same way, and `seedproof diff` the
two records.
