# autoresearch-arch

Autonomous architecture search experiment built on top of `train_gpt.py`.
The agent explores novel model architectures to minimize **val_bpb**
under a strict **≤ 16M parameter** budget.

---

## Default Architecture Overview

The baseline model in `train_gpt.py` is a GPT with the following structure:

- **Embedding**: learned token embedding (`vocab_size=1024`), optionally tied
  with the output head (`tie_embeddings=True` by default, saves ~0.5M params)
- **Attention**: Grouped-Query Attention (GQA) with `num_kv_heads < num_heads`.
  Q and K are RMSNorm'd, then RoPE is applied, then a per-head learned `q_gain`
  scales Q before `F.scaled_dot_product_attention` with causal mask.
- **MLP**: relu² activation — `relu(fc(x))² → proj(·)`. Two `CastedLinear` layers.
- **Block residual**: learned `resid_mix` blends the current hidden state with
  the original embedding (`x = mix[0]*x + mix[1]*x0`), then adds attention
  (scaled by `attn_scale`) and MLP (scaled by `mlp_scale`).
- **U-Net skip connections**: the first `num_layers // 2` blocks store
  activations; the last `num_layers // 2` blocks add them back via learned
  `skip_weights`.
- **Output**: logit softcapping — `softcap * tanh(logits / softcap)`.
- **Precision**: model body runs in **bf16**; `CastedLinear` stores weights
  in **fp32** and casts to bf16 at forward time; control params (`attn_scale`,
  `mlp_scale`, `resid_mix`, `q_gain`, `skip_weights`) stay in fp32.
- **Compilation**: the model is compiled with `torch.compile`. Architectural
  changes must be compatible with torch.compile (avoid unsupported ops or
  excessive dynamic control flow).
- **Distributed**: training uses `DistributedDataParallel` (DDP) across 8 GPUs
  with `grad_accum_steps = 8 // world_size`.

Default hyperparameters:

| Field | Default |
|---|---|
| `num_layers` | `9` |
| `model_dim` | `512` |
| `num_heads` | `8` |
| `num_kv_heads` | `4` |
| `mlp_mult` | `2` |
| `tie_embeddings` | `True` |
| `rope_base` | `10000.0` |
| `logit_softcap` | `30.0` |
| `qk_gain_init` | `1.5` |

This default config is ~17M params and exceeds the 16M budget. You must shrink
the model before the first run.

---

## Setup

Work with the user to complete the following before starting experiments:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`).
   The branch `autoresearch/<tag>` must not already exist.

2. **Create the branch**:
   ```shell
   git checkout -b autoresearch/
   ```

3. **Read the in-scope files** for full context:
   - `train_gpt.py` — the only file you modify. Contains the GPT model,
     optimizer, and training loop.
   - `records/` — past competitive submissions. Browse the READMEs for
     architectural inspiration (note: many submissions also change the
     optimizer or quantization, which is off-limits here).

4. **Verify data exists**: Check that the following paths exist:
   - `./data/datasets/fineweb10B_sp1024/` — training and validation shards
   - `./data/tokenizers/fineweb_1024_bpe.model` — SentencePiece tokenizer

   If missing, tell the human to run the data preparation script first.

5. **Estimate baseline parameter count**: The default config is approximately
   17M params and already exceeds the 16M budget. You must shrink the model
   before the first run. Use this quick estimate formula:
   ```
   num_skip_weights = num_layers // 2

   total ≈ vocab_size × model_dim                              # embedding (shared if tied)
         + num_layers × (
             model_dim × model_dim                              # Q proj
           + model_dim × (num_kv_heads / num_heads × model_dim) # K proj
           + model_dim × (num_kv_heads / num_heads × model_dim) # V proj
           + model_dim × model_dim                              # attn out proj
           + model_dim × (mlp_mult × model_dim)                 # MLP fc
           + (mlp_mult × model_dim) × model_dim                 # MLP proj
           + ~3 × model_dim                                     # control params (attn_scale, mlp_scale, resid_mix)
           + num_heads                                           # q_gain (per head)
         )
         + num_skip_weights × model_dim                         # U-Net skip connections
   ```

   Always verify the actual count from `grep "^model_params:" run.log`
   after each run.

6. **Initialize results.tsv**: Create `results.tsv` with just the header row.

7. **Confirm with the user and begin**.

---

## The 16M Parameter Rule

This is the **hardest constraint** in this experiment. Every run must satisfy:

`total_params ≤ 16,000,000`

**Check param count immediately after each run**:
```bash
grep "^model_params:" run.log
```

If a run exceeds 16M params, treat it as a crash:

- Log status as crash, description as `param_budget_exceeded`
- `git reset --hard HEAD~1`
- Continue to the next experiment

Tie embeddings (`tie_embeddings = True`) is strongly recommended.
It saves ~0.5M params for free. Keep it on unless you have a strong reason.

## What You CAN Modify

Only modify the model architecture and model-shape hyperparameters inside
`train_gpt.py`. Specifically:

In the `Hyperparameters` class — model shape fields only:
```python
num_layers     = 9        # number of transformer blocks
num_kv_heads   = 4        # GQA kv heads (must divide num_heads evenly)
model_dim      = 512      # hidden dimension
num_heads      = 8        # attention heads (must divide model_dim evenly)
mlp_mult       = 2        # MLP hidden expansion factor
tie_embeddings = True     # share embedding and lm_head weights (saves ~0.5M params)
rope_base      = 10000.0  # RoPE base frequency
logit_softcap  = 30.0     # logit soft-capping value
qk_gain_init   = 1.5      # initial QK gain value
```

`tied_embed_init_std` (default `0.005`) controls embedding initialization and
is also considered a modifiable model parameter (not an optimizer parameter).

Do NOT touch `vocab_size` — it must remain 1024 to match the fixed tokenizer.

Model architecture classes — anything inside:
- `GPT` — overall forward pass, U-Net skips, embedding, final norm, logits
- `Block` — residual structure, normalization placement, skip weights
- `CausalSelfAttention` — attention mechanism, QK norm, RoPE usage
- `MLP` — activation function (default: relu²), gating, hidden size
- `CastedLinear` — linear layer with fp32 weight storage, bf16 compute
- `RMSNorm` — RMS normalization
- `Rotary` — RoPE cos/sin cache
- `apply_rotary_emb` — RoPE application helper

You may add new `nn.Module` subclasses or helper functions to support
architectural changes. All changes must be compatible with `torch.compile`.

Do NOT modify `train_gpt_mlx.py` — it is a separate MLX variant and is not
part of this experiment.

## What You CANNOT Modify

- **Optimizer code**: Do NOT change `Muon`, `zeropower_via_newtonschulz5`,
  or any optimizer hyperparameters in `Hyperparameters`:
  `embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr`, `scalar_lr`,
  `muon_momentum`, `muon_backend_steps`, `muon_momentum_warmup_start`,
  `muon_momentum_warmup_steps`, `beta1`, `beta2`, `adam_eps`,
  `grad_clip_norm`, `warmdown_iters`, `warmup_steps`.
  The optimizer is completely fixed.

- **Training hyperparameters**: Do NOT change `train_batch_tokens`,
  `train_seq_len`, `val_batch_size`, `val_loss_every`, `iterations`,
  `train_log_every`, `max_wallclock_seconds`, or `seed`.

- **Training loop**: Do NOT change gradient accumulation, LR schedule,
  warmup/warmdown logic, or wallclock stopping logic.

- **Data pipeline**: Do NOT change `TokenStream`, `DistributedTokenLoader`,
  `load_data_shard`, or any data loading code.

- **Evaluation**: Do NOT change `eval_val`, `build_sentencepiece_luts`,
  `load_validation_tokens`, or any validation metric code.

- **Serialization**: Do NOT change quantization or model saving code.

- **vocab_size**: Must remain 1024.

- **Third-party libraries**: Do NOT import or install anything beyond what is
  already used in the existing code. Permitted imports:
  - Python standard library (`math`, `os`, `time`, `sys`, `random`,
    `copy`, `glob`, `io`, `uuid`, `zlib`, `subprocess`, `pathlib`, etc.)
  - `torch` and all its submodules (`torch.nn`, `torch.nn.functional`,
    `torch.distributed`, `torch.compile`,
    `F.scaled_dot_product_attention`, `triton`, etc.)
  - `numpy`
  - `sentencepiece`

  Explicitly forbidden: `flash_attn`, `fa3`, `xformers`, `apex`, `deepspeed`,
  or any package requiring a separate `pip install`.

## Running an Experiment

Launch training with:
```shell
torchrun --standalone --nproc_per_node=8 train_gpt.py > run.log 2>&1
```

Time budget: Each run is capped at 10 minutes wall clock
(`MAX_WALLCLOCK_SECONDS=600`).

Extract results after each run:
```shell
# Check param count (MUST be ≤ 16M)
grep "^model_params:" run.log

# Primary metric — use the last printed val_bpb
grep "^step.*val_bpb" run.log | tail -5

# Memory usage
grep "^peak memory" run.log

# Detect crashes or early stops
grep "stopping_early\|Error\|Traceback" run.log | head -20
```

The `val_bpb` to record is the `final_int8` model `val_bpb` printed before the run ends.

## Output Format

A successful run prints lines like:

```
model_params:14500000
step:1000/20000 val_loss:2.3298 val_bpb:1.3798 train_time:80658ms step_avg:80.66ms
...
step:7395/20000 val_loss:2.0641 val_bpb:1.2225 train_time:600035ms step_avg:81.14ms
stopping_early: wallclock_cap train_time:600035ms step:7395/20000
peak memory allocated: 20241 MiB reserved: 20390 MiB
Serialized model: 67236226 bytes
Code size: 49619 bytes
Total submission size: 67285845 bytes
Serialized model int8+zlib: 15815816 bytes (payload:17189152 raw_torch:17234444 payload_ratio:3.91x)
Total submission size int8+zlib: 15865435 bytes
final_int8_zlib_roundtrip val_loss:2.0737 val_bpb:1.2282 eval_time:2544ms
final_int8_zlib_roundtrip_exact val_loss:2.07374725 val_bpb:1.22818993
```

## Logging Results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header and columns:
```
commit	val_bpb	params_M	memory_mib	status	description
```

- `commit` — git short hash (7 chars)
- `val_bpb` — final val_bpb (use `0.000000` for crashes)
- `params_M` — model params in millions, 1 decimal (e.g. `14.5`). Use `0.0` for crashes
- `memory_mib` — peak memory allocated in MiB from the log. Use `0` for crashes
- `status` — `keep`, `discard`, or `crash`
- `description` — short description of what changed

Example:
```
commit	val_bpb	params_M	memory_mib	status	description
a1b2c3d	0.000000	17.0	0	crash	param_budget_exceeded: default config
b2c3d4e	0.912300	14.2	9800	keep	baseline: layers=8 dim=448 mlp_mult=2
c3d4e5f	0.908100	15.1	10200	keep	deeper+narrower: layers=11 dim=384
d4e5f6g	0.915000	14.8	9900	discard	relu2->gelu activation
e5f6g7h	0.000000	14.8	0	crash	OOM: added extra attention layer
```

Do not commit `results.tsv` — leave it untracked by git.

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

**LOOP FOREVER:**

1. Check git state: confirm current branch and latest commit.

2. Design an architectural change. Think carefully about the 16M budget
   before touching any code. Use the param count formula to pre-estimate.

3. Modify `train_gpt.py` with the architectural change only.

4. `git commit` with a short descriptive message.

5. Run the experiment:
   ```shell
   torchrun --standalone --nproc_per_node=8 train_gpt.py > run.log 2>&1
   ```

6. Check param count first:
   ```shell
   grep "^model_params:" run.log
   ```

   If > 16,000,000: log as crash with `param_budget_exceeded`,
   `git reset --hard HEAD~1`, continue to next idea.

7. Read the results:
   ```shell
   grep "^step.*val_bpb" run.log | tail -3
   grep "^peak memory" run.log
   grep "stopping_early\|Traceback" run.log | head -5
   ```

8. If output is empty or shows Traceback: run crashed.
   ```shell
   tail -n 50 run.log
   ```
   Fix if trivial (typo, shape mismatch). Otherwise log as crash and
   `git reset --hard HEAD~1`.

9. Record in `results.tsv`.

10. Keep or discard:
    - val_bpb improved (strictly lower) → keep the commit, advance branch
    - val_bpb same or worse → `git reset --hard HEAD~1`

## Common Pitfalls

- `num_heads` must evenly divide `model_dim` (otherwise attention reshape fails)
- `num_kv_heads` must evenly divide `num_heads` (GQA repeat requirement)
- U-Net skip connections assume `num_layers` is even (`num_layers // 2` skips
  on each side). Odd layer counts still work but the middle layer has no skip.
- `torch.compile` does not support all operations. If you add custom ops or
  heavy dynamic control flow, training may fail to compile. Test early.
- Changing `mlp_mult` or `model_dim` has a large impact on param count —
  always re-estimate before running.
- OOM: increasing `model_dim` or adding parameters increases memory. Check
  `peak memory` in the log if runs crash silently.

## Simplicity Criterion

All else being equal, simpler is better.

- A small val_bpb improvement that adds significant complexity: probably not worth keeping.
- Removing a component and getting equal or better val_bpb: always keep — that is a simplification win.
- A near-zero improvement from a clean, well-motivated change: keep it.
- A near-zero improvement from a hacky workaround: discard it.

## Research Strategy

You are constrained to architecture only. The optimizer and data are fixed.
Your only lever is the model structure and its hyperparameters.

Recommended exploration order (roughly easy to ambitious):

**Phase 1 — Establish a valid baseline**

Find a config with ≤ 16M params that trains stably. The default config is ~17M,
so shrink it. Suggested starting point:
```python
num_layers = 8
model_dim = 448
num_heads = 8
num_kv_heads = 4
mlp_mult = 2
tie_embeddings = True
```

This should be around 13-14M params. Confirm with the log.

**Phase 2 — Depth vs width sweep**

Under 16M, try different (`num_layers`, `model_dim`) combinations at roughly
equal total param count:

- Deeper + narrower: e.g. `layers=12`, `dim=352`
- Shallower + wider: e.g. `layers=6`, `dim=512`
- Asymmetric U-Net: e.g. unequal encoder/decoder halves

**Phase 3 — Attention variants**

- GQA ratio: try `num_kv_heads` = 1, 2, 4, 8
- RoPE base: try 1000, 10000, 100000
- QK gain init: try 1.0, 1.5, 2.0
- Remove QK RMSNorm (may destabilize — test carefully)
- NoPE: remove RoPE on some layers entirely

**Phase 4 — MLP variants**

The default MLP uses **relu²** (relu squared): `relu(fc(x))² → proj(·)`.

- Activation: replace relu² with SwiGLU-style gating
  (note: gated MLP uses ~1.5× params of standard MLP for same hidden size,
  shrink `model_dim` accordingly to stay under 16M)
- `mlp_mult`: try 1, 3, 4
- Asymmetric MLP: different expansion per layer

**Phase 5 — Residual and skip structure**

- The existing U-Net skip: try removing it entirely
- Try different `skip_weight` initializations
- Try learned vs fixed mixing in `resid_mix`
- Try pre-norm vs post-norm placement
- Try placing norm only on one branch

**Phase 6 — Embedding and output**

- `logit_softcap`: try 10, 20, 50
- Embedding scaling: multiply embedding output by a learned scalar
- `tied_embed_init_std` (default `0.005`): try different values — this is
  a model initialization parameter and is allowed to change

**Phase 7 — Novel combinations**

Combine the best individual findings. If deeper+narrower won in Phase 2
and SwiGLU won in Phase 4, try them together.

## Key Reminders

1. **16M param hard limit**: check `model_params` in every log before recording
2. **vocab_size = 1024**: never change this
3. **No new libraries**: only `torch`, `numpy`, `sentencepiece`, stdlib
4. **Only architecture**: never touch optimizer, training loop, or evaluation
5. **NEVER STOP**: do not pause to ask the human whether to continue.
   Once the loop has started, run indefinitely until manually interrupted.
   If you run out of ideas, re-read the research strategy, look at which
   near-misses came closest, and try combining them or pushing them further.
   The loop runs until the human stops you, period.
