# Bonsai narrator server setup

## What it is and why we have it

The auto-grader's "Project Paint Dry" narrator (the live commentary
sidecar that produces the per-item topic lines and the post-game
wrap-up) is driven by **Bonsai-8B-mlx-1bit**, a 1-bit ternary
quantization of Qwen3-8B published by PRISM ML. It fits in ~1.3 GB
of RAM, runs at ~29 tok/s on Apple Silicon, and is used as the
narrator model because it's cheap enough to run alongside the
much-larger grader model on the same machine without resource
contention.

The narrator code lives in `auto_grader/thinking_narrator.py` and
defaults to talking to the bonsai server at `http://localhost:8001`
with the served model name `Bonsai-8B-mlx-1bit`. Both can be
overridden on the smoke command line:

```sh
uv run python scripts/smoke_vlm.py \
  --narrate \
  --narrator-url http://localhost:8001 \
  --narrator-model Bonsai-8B-mlx-1bit \
  ...
```

## Bonsai is independent from the grader server

There are **two OMLX servers** in play during a typical eval:

| server                       | port             | model(s)                       | role           |
|------------------------------|------------------|--------------------------------|----------------|
| local bonsai                 | `localhost:8001` | `Bonsai-8B-mlx-1bit`           | live narrator  |
| remote grader (the big box)  | `macbook-pro-2.local:8001` | `qwen3p5-35B-A3B`, `gemma-4-26b-a4b-it-bf16`, `Qwen3p5-122B-A10B-mlx-mixedp`, `step-3p5-flash-*`, etc. | the actual grader |

They share a port number but live on different hosts, so they don't
collide. The narrator client uses `--narrator-url` (default
`http://localhost:8001`); the grader client uses `--base-url` (default
`http://macbook-pro-2.local:8001` since the
[`d439c0f`](#) "Default --base-url to mDNS" commit, so the grader
URL automatically follows DHCP renewals on the big box).

## Why bonsai needs the PRISM-patched mlx-openai-server, not stock OMLX

Stock MLX (as of 0.31.1) does **not** support `bits=1` quantization.
PRISM ML maintains an MLX fork with Metal dequantization kernels for
1-bit affine quantization. That fork is built from source and
installed into a `mlx-openai-server` uv-tool environment, which
becomes the launcher for bonsai.

Trying to launch bonsai with the upstream `omlx serve` binary (the
nice menu-bar app) will fail because the underlying MLX install
doesn't have the PRISM kernels. **Use `mlx-openai-server launch`,
not `omlx serve`, for the bonsai narrator.**

(The grader server on the big box uses the standard OMLX stack with
non-1-bit models, so it doesn't need the PRISM fork. Different
machine, different setup.)

## How to start bonsai locally

```sh
~/.local/bin/mlx-openai-server launch \
  --model-path prism-ml/Bonsai-8B-mlx-1bit \
  --model-type lm \
  --port 8001 \
  --host 127.0.0.1 \
  --served-model-name Bonsai-8B-mlx-1bit
```

Notes on the flags:

- `--model-path prism-ml/Bonsai-8B-mlx-1bit` resolves from the local
  Hugging Face cache at
  `~/.cache/huggingface/hub/models--prism-ml--Bonsai-8B-mlx-1bit/`.
  If the cache is missing the model, the first run downloads it
  (~1.3 GB).
- `--model-type lm` is text-only; bonsai doesn't have a vision tower.
- `--port 8001` matches the narrator client default.
- `--served-model-name Bonsai-8B-mlx-1bit` is the name returned by
  `/v1/models` and accepted in request `model:` fields. It must
  match `_DEFAULT_NARRATOR_MODEL` in `thinking_narrator.py`
  (currently `Bonsai-8B-mlx-1bit`).

To run as a backgrounded process and capture logs:

```sh
~/.local/bin/mlx-openai-server launch \
  --model-path prism-ml/Bonsai-8B-mlx-1bit \
  --model-type lm \
  --port 8001 \
  --host 127.0.0.1 \
  --served-model-name Bonsai-8B-mlx-1bit \
  > /tmp/bonsai-server.log 2>&1 &
```

## How to verify it's up

```sh
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

You should see exactly one model registered: `Bonsai-8B-mlx-1bit`.
A successful health check returns within a few hundred milliseconds.

If `curl` hangs or returns connection refused, the server isn't
running. Check `/tmp/bonsai-server.log` (or wherever you redirected
stderr) for the failure mode. The most common ones:

- **Port 8001 already in use** — another mlx-openai-server instance
  or some other process is squatting on the port. `lsof -i :8001`
  will tell you what.
- **Model not in HF cache and no network** — first launch needs to
  download ~1.3 GB from HuggingFace.
- **MLX version mismatch** — the uv-tool environment has stock MLX
  instead of the PRISM fork. Server starts but model load fails with
  an "unsupported quantization bits=1" error. Reinstall via project
  pocket pilfer (see References).

## Symptoms of bonsai being down vs. up

When bonsai is down, the auto-grader continues to run — the narrator
just can't produce live commentary. Concretely:

- `--narrate` runs will print the per-item header but no topic lines
  (the narrator thinks "writing..." forever and dispatches all fail
  silently).
- The `narrator.jsonl` log will have `header` events but no `delta`
  or `commit` events.
- The post-game wrap-up will be missing or print a stale partial.
- The grader itself is unaffected — it talks to the big box at
  `macbook-pro-2.local:8001`, not to bonsai. **Predictions still
  land in `predictions.jsonl` correctly.**

So a missing narrator is a quality-of-life issue, not a data-loss
issue. You'll notice it because the live Project Paint Dry display
is empty mid-run.

## Killing a stuck server

```sh
pkill -f "mlx-openai-server.*Bonsai-8B-mlx-1bit"
```

Or find the PID and `kill` it directly:

```sh
lsof -i :8001 | grep LISTEN
kill <PID>
```

## References

- **Project Pocket Pilfer** — the original epistaxis project that
  did the PRISM MLX build-from-source work. See
  `~/dev/epistaxis/projects/omlx/epistaxis.md § Project Pocket Pilfer`
  for the historical setup notes including which PRISM commit the
  uv-tool env was built from.
- **PRISM ML's MLX fork** — `PrismML-Eng/mlx`, branch
  `1bit-affine-quantization`. The fork that the
  `mlx-openai-server` uv-tool env is patched against.
- **Bonsai model card** — `prism-ml/Bonsai-8B-mlx-1bit` on
  HuggingFace. 1-bit ternary quantization of Qwen3-8B.
- **OMLX upstream** — `~/dev/omlx/`. The stock server the rest of
  the stack uses. Doesn't support bits=1, hence the divergence.

## Open questions / known gaps

- The bonsai server is **not** managed by `brew services` or any
  launchd plist; it has to be manually started after each reboot.
  Worth wiring up as a launchd plist if it's a regular pain point.
- We don't currently have a health-check / auto-restart wrapper.
  When the server stalls (which happens occasionally — see the
  smoke run on 2026-04-08 16:43 where it died mid-stream), the
  smoke client just waits on a wedged TCP connection until the
  600s urlopen timeout fires. A liveness probe in the smoke loop
  would be a small improvement.
- The bonsai server on `macbook-pro-2.local` (the big box) registers
  models named `bonsai` and `pism-ml-bonsai-8b-mlx-1bit`, not
  `Bonsai-8B-mlx-1bit`. So if you ever want to point the narrator
  at the big box's bonsai instead of running it locally, you'll
  need `--narrator-url http://macbook-pro-2.local:8001 --narrator-model bonsai`.
