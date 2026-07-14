# The seedproof run-record format

A *run record* is one recorded generation as a single JSON file. Version 1
is identified by the top-level key `"seedproof_record": 1`; a file without
that key is rejected, and a file with a higher version than the tool
supports asks you to upgrade instead of guessing.

Records are written with sorted keys, two-space indentation, UTF-8, a
trailing newline, and an atomic rename — so they diff cleanly in git and a
crashed write never leaves a truncated record behind.

## Top-level fields

| Field | Required | Meaning |
|---|---|---|
| `seedproof_record` | yes | format version; `1` for this document |
| `prompt` | yes | the full prompt the run answered |
| `prompt_sha256` | written | hex digest of `prompt`; verified on load, so an edited record fails loudly |
| `config` | yes | the runtime configuration object (below) |
| `tokens` | yes | the emitted token stream, in order (below) |
| `name` | no | human label; defaults to the filename stem on load |
| `created_at` | no | informational timestamp string; never compared |
| `notes` | no | free-form text; never compared |

## `config`

| Field | Type | Default | Compared as |
|---|---|---|---|
| `model` | string | `""` | model identity axis |
| `backend` | string | `""` | runtime axis (e.g. `cpu`, `cuda`, `metal`, `vulkan`) |
| `device` | string | `""` | runtime axis (e.g. `gpu0`) |
| `quant` | string | `""` | runtime axis (e.g. `fp16`, `q8_0`, `q4_k_m`) |
| `seed` | int or null | `null` | sampling axis; only meaningful when sampling is stochastic |
| `sampler` | string | `"greedy"` | sampling axis |
| `temperature` | number | `0.0` | sampling axis |
| `top_k` | int | `0` | sampling axis |
| `top_p` | number | `1.0` | sampling axis |
| `extra` | object | `{}` | free-form runtime knobs; compared verbatim as `extra.<key>` runtime axes |

Unknown keys inside `config` are an error — put custom knobs under `extra`
so a typo (`quanto`) cannot silently create an uncompared field.

A run counts as **stochastic** when `sampler != "greedy"` and
`temperature > 0`; only then can a seed difference explain a divergence.

## `tokens[]`

| Field | Required | Meaning |
|---|---|---|
| `text` | yes | the token's surface text |
| `id` | no | tokenizer id; enables the stronger `id` comparison basis |
| `logprob` | no | log-probability of the chosen token (nats) |
| `top` | no | top-k candidates: `{ "text", "logprob", "id"? }` each |

The more optional fields your capture keeps, the stronger the forensics:

- `id` on every token → comparisons use ids (`auto` basis), which
  distinguishes a tokenizer/vocab mismatch from a behavior change;
- `logprob` on every token → shared-prefix drift metrics;
- `top` at every step → near-tie analysis at the divergence point, which is
  what separates "a numerical tie-break" from "the distribution moved".

## Comparison basis

Two streams are compared position-by-position on a **basis**:

- `id` — token ids; the ground truth when both runs captured them;
- `text` — token texts; the fallback, and the only option for captures
  without ids (e.g. plain SSE streams);
- `auto` (default) — `id` when both runs have full ids, else `text`.

Stream fingerprints (`seedproof ls`, matrix class labels) are 12-hex-digit
SHA-256 digests of the stream on the stated basis; `ls` always prints the
text-basis fingerprint so every record, with or without ids, is comparable
at a glance.
