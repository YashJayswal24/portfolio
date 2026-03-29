---
layout: post
title: "Scaling LLMs: MoE Routing & JAX Parallelism on TPU"
date: 2026-03-29 19:00:00 +0900
description: A deep dive into JAX's three parallelism modes — from auto-sharding to manual shard_map — implementing Mixture-of-Experts routing with a 43× speedup on TPU v5e-8.
tags: jax tpu systems-engineering llm scaling moe parallelism
categories: engineering
giscus_comments: false
related_posts: true
---

_Up Next_ was the promise at the end of [Part 9 — TPU Profiling](/blog/2026/tpu-profiling). We knew _what_ to profile. Now we learn _how to write code that gives the compiler no room to make bad decisions._

This is Part 10 of my journey through [_How To Scale Your Model_](https://jax-ml.github.io/scaling-book/jax-stuff/) — and it's the most hands-on chapter yet. The full working code lives in [`moe-scaling-on-tpu.ipynb`](https://www.kaggle.com/code/reptor420/moe-scaling-on-tpu/notebook) (Kaggle, TPU v5e-8), and the implementation folder is at [`10_jax_parallelism_moe/`](https://github.com/YashJayswal24/Model_scaling_jax/tree/main/10_jax_parallelism_moe).

---

## The Three Modes of JAX Parallelism

JAX gives you three different "contracts" with the compiler:

| Mode         | API                       | Who decides communication?            |
| ------------ | ------------------------- | ------------------------------------- |
| **Auto**     | `jax.jit` + Auto axes     | XLA/Shardy compiler                   |
| **Explicit** | `jax.jit` + Explicit axes | JAX type system (errors on ambiguity) |
| **Manual**   | `jax.shard_map`           | **You**                               |

The book frames this beautifully: modes 1 and 2 let you write single-device code and trust the system to scale it. Mode 3 — `shard_map` — hands you a **local, per-device view** of the array, and every byte of communication is your responsibility.

That sounds scary. But once you see what the compiler does wrong on a real MoE model, you'll be grateful for the control.

---

## Problem 1: Sharded Average & Roll Difference

**Mesh setup:** 8 TPUs arranged as `(x=2, y=4)`, `AxisType.Explicit`.

The first exercise: given an array `A: float32[S_X, D_Y]` sharded across the mesh, compute the mean _within each (X, Y) shard_ — resulting in an `[X, Y]` output with no cross-device communication.

### With `jax.jit`

The trick is to reshape the array to expose the shard boundaries, then take the mean over the local (non-sharded) axes:

```python
@jax.jit
def average(arr):
    # Reshapes [x, p, y, q] → mean over (1, 3) gives [x, y]
    arr = jnp.mean(arr, axis=(1, 3))
    return jax.lax.with_sharding_constraint(arr, P('x', 'y'))
```

Output: `[[4.5, 6.5, 8.5, 10.5], [20.5, 22.5, 24.5, 26.5]]` ✓  
**Zero communication.** The `with_sharding_constraint` is just a guard — no AllGather is needed since each device independently owns its shard data.

### With `shard_map`

Even cleaner. Each device sees only its local `[p, q]` block and returns a scalar:

```python
@jax.shard_map(in_specs=jax.P('x','y'), out_specs=jax.P('x','y'))
def avg_shard_map(arr):
    assert arr.shape == (p, q)
    return jnp.mean(arr).reshape((1, 1))
```

Same output, same zero-communication guarantee — but now the device-local view makes this _obvious_ from the code rather than implicit in the compiler.

For the roll-difference function (`roll(x, shift, axis=0) - x` within each X shard), `shard_map` makes the intent equally clear:

```python
@jax.shard_map(in_specs=(jax.P('x','y'), jax.P('x','y'), jax.P()), out_specs=jax.P('x','y'))
def roll_diff(base, roll, shift):
    return base - jnp.roll(roll, shift=shift, axis=0)
```

Result with `shift=1`: rows alternate between `-8` and `+8` — exactly the intra-shard difference expected. ✓

---

## Problem 2: Mixture of Experts — The 43× Speedup

This is the centerpiece. We have:

- `W: float32[E, D, F]` — 8 expert weight matrices
- `A: float32[S, D]` — 2048 token activations
- `B: int32[S]` — routing assignments (which expert processes each token)

Goal: `Out[i] = W[B[i]] @ A[i]` — route each token to exactly one expert.

**Dimensions (Llama-proxy scale):** `E=8, S=2048, D=4096, F=14336` on 8 TPU cores.

### Naive `jit`: 73 ms

The natural scan-based implementation:

```python
def process_exp(carry, e):
    mask = (B == e)[:, None]   # [S, 1] — which tokens belong to expert e
    to_add = A @ W[e]           # [S, F] — ALL tokens × expert e's weights
    return carry + to_add * mask, None  # Throw away E-1 results

out, _ = jax.lax.scan(process_exp, output, jnp.arange(E))
```

When I profiled this with `jax.profiler.trace`, XLA had made a decision I didn't want: it sharded _both_ `A` and `W` along `x` (data-parallel). Every device had to gather the full activation matrix before running its experts. **Latency: 73 ms.** A disaster at production scale.

### Pipelined All-to-All: 1.7 ms (43× faster)

The fix is explicit routing: instead of gathering _all_ activations to every device, use **All-to-All** to send only the tokens that belong to each expert.

The algorithm (device-local view via `shard_map`):

```
1. Sort local tokens by their assigned expert → [S_x] sorted
2. Pack C tokens per expert into send buffer → [E, C, D]
3. All-to-All forward: send to expert-owning devices → [E_x, N*C, D]
4. Local dense matmul (zero padding waste) → [E_x, N*C, F]
5. All-to-All backward: return results → [E, C, F]
6. Scatter back to original token positions → [S_x, F]
```

To enable compute-communication overlap, the loop is **unrolled in Python** (not `lax.scan`), which lets XLA flatten the execution graph and pipeline the next network fetch behind the current matmul:

```python
for i in range(num_scan_steps):
    # 1. Math first (keeps MXU busy)
    carry_out = compute_and_scatter(recv_A_curr, valid_curr, safe_idx_curr, carry_out)
    # 2. Network second (runs in background while math executes)
    recv_A_curr, valid_curr, safe_idx_curr = fetch_chunk(i + 1)
```

**Latency: 1.7 ms.** A **43× speedup** over the naive implementation.

The beauty here: with `check_vma=True` on the `shard_map`, JAX _verifies_ that no accidental communication was inserted. The only network traffic is the two explicit All-to-All calls.

### Top-k Routing (k=2): Zero Overhead

Extending to Top-2 routing (each token processed by 2 experts and averaged) adds zero latency:

```python
flat_B = B_local.reshape(-1)           # [S_x, 2] → [S_x * 2]
flat_A = jnp.repeat(A_local, k, axis=0) # Duplicate each token k times
# ... same pipeline ...
final_out = jnp.mean(final_out_k, axis=1)  # Average k expert outputs
```

**Latency: ~1.7 ms.** Same as Top-1. The pipeline handles double the tokens at the same capacity.

---

## Problem 3: Collective Matmuls & Overlapped MLP

The book references [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959) on collective matmuls — the idea of overlapping inter-chip communication with local computation. We implement three variants:

### AllReduce Matmul

`A[BX, DY] @ W[DY, F] → Out[BX, F]` — contracting across the sharded `D` dimension:

```python
@jax.shard_map(in_specs=(P('x','y'), P('y',None)), out_specs=P('x',None))
def compute_matmul_allreduce(A, W):
    return jax.lax.psum(A @ W, axis_name='y')
```

Each device computes a partial sum over its local `D/Y` slice, then `psum` AllReduces across the `y` axis.

### ReduceScatter — Three Ways

For the down-projection `Tmp[BX, FY] @ W2[FY, D] → Out[BX, DY]`:

**JAX built-in `psum_scatter`:** ~1.01 ms — the baseline.

**Manual ring (unidirectional, y-1 steps):** ~1.2 ms. Counterintuitively _slower_ — XLA's built-in has hardware-level stream fusion we can't replicate.

**Recursive halving (bidirectional, log₂y steps):**

```python
# At each step, exchange half the array with XOR partner
perm = tuple((j, j ^ (1 << step)) for j in range(y))
recv_data = jax.lax.ppermute(send_data, axis_name='y', perm=perm)
state = keep_data + recv_data  # Array shrinks by 2× each step
```

This is **bidirectional ring** — device `i` always exchanges with `i XOR 2^step`. After `log₂(y)` hops, each device holds exactly its fully-summed `D/Y` slice. Latency: ~1.1 µs — within spitting distance of `psum_scatter`.

### End-to-End Overlapped Transformer MLP

Composing AllGather up-projection + recursive halving ReduceScatter down-projection:

```
In[BX, DY] → AllGather(y) → [BX, D] → Win[D, FY] → GeLU → recursive_halving_RS → Out[BX, DY]
```

**Latency: ~1.25 ms** — parity with `jax.jit` baseline (~1.2 ms). The overlap provides _no additional speed here_, confirming the book's note: XLA already handles this internally. But what we gain is **transparency** — every byte of communication is visible in the Perfetto trace.

---

## What I Learned

**1. The compiler can be confidently wrong.**  
Auto-sharding on the naive MoE chose data-parallel weight sharding. This triggered an AllGather of activations — the most expensive possible communication pattern. Without `shard_map` + explicit All-to-All, we'd have never caught this.

**2. Python loops are a feature, not a bug.**  
Using a Python `for` loop (not `lax.scan`) inside `shard_map` unrolls the computation graph. XLA sees the full static schedule and can pipeline communication behind computation. `lax.scan` would collapse this into a dynamic loop that prevents overlap.

**3. Bidirectional != always faster.**  
Recursive halving (bidirectional) achieved ~1.1 µs, matching `psum_scatter` but not beating it. The hardware-optimized CollectiveReduceScatter knows things about the TPU interconnect topology that we don't. Use built-ins first; go manual only when the profiler shows waste.

**4. Expert capacity `C` is a real knob.**  
Setting `C = 2 × (S / (E × N))` gave us ~8 communication rounds. Too small → more network round-trips. Too large → wasted padding compute. Profiling both extremes is the right way to tune.

---

## Performance Summary

| Experiment                   | Latency    | Key Takeaway                             |
| ---------------------------- | ---------- | ---------------------------------------- |
| Naive `jit` MoE              | 73 ms      | AllGather on activations is catastrophic |
| **Pipelined All-to-All MoE** | **1.7 ms** | **43× speedup** with explicit routing    |
| Top-k MoE (k=2)              | 1.7 ms     | Zero overhead over Top-1                 |
| `jit` Auto MLP               | 1.2 ms     | XLA baseline                             |
| `psum_scatter` ReduceScatter | 1.01 ms    | Best built-in option                     |
| Recursive halving RS         | ~1.1 µs    | Near-optimal manual variant              |
| Overlapped Transformer MLP   | 1.25 ms    | Transparent comm, JIT-speed parity       |

**Hardware:** Kaggle TPU v5e-8 (8 cores) · **Total runtime:** 72 seconds · **9 AOT-compiled profiler traces**

---

## Code & Resources

- 📓 **Kaggle notebook:** [reptor420/moe-scaling-on-tpu](https://www.kaggle.com/code/reptor420/moe-scaling-on-tpu/notebook)
- 📁 **Implementation folder:** [`10_jax_parallelism_moe/`](https://github.com/YashJayswal24/Model_scaling_jax/tree/main/10_jax_parallelism_moe)
- 📖 **Book section:** [Part 10 — All About JAX](https://jax-ml.github.io/scaling-book/jax-stuff/)
- 🔗 **Full repo:** [YashJayswal24/Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

---

_Part 10 complete. The scaling book series is done — final conclusions live in [Part 11](https://jax-ml.github.io/scaling-book/conclusion)._
