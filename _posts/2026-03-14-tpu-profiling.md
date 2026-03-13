---
layout: post
title: "TPU Profiling: When Math Meets Reality"
date: 2026-03-14 09:00:00 +0900
description: Digging into JAX profiles, HLO traces, and observing our theoretical FLOP scaling limits in action.
tags: jax tpu profiling xla compilers optimization
categories: engineering
giscus_comments: false
related_posts: true
---

Everything leading up to Part 9 has been theoretical. We built formulas for bandwidth limits, FLOP constraints, and expected batch times.

But how do we know if our models are actually hitting these limits? The answer lies in the **JAX Profiler** and the **XLA trace viewer**.

---

## 🔍 The Stack: JAX → StableHLO → LLO → IMEM

When you run `y = jnp.dot(x, W)` on a TPU, JAX traces your computation and compiles it down several levels. Performance debugging happens at the **HLO (High Level Optimizer)** level.

Reading an HLO operation gives you immediate insights into memory access inefficiencies. For example:
```text
bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}
```
- **Shape & Type:** `bfloat16` tensor, `[32, 32, 8192]`.
- **Layout (`{2,1,0}`)**: Data dimension ordering (critical for systolic array throughput).
- **Tiling (`T(8,128)(2,1)`)**: The physical block chunks memory is padded into. Poor tiling can inflate memory sizes by 1.6x.
- **Space (`S(1)`)**: VMEM (Virtual Memory / fast compute memory) vs `S(0)` HBM (High Bandwidth Memory).

---

## ⏱️ Predicting the Future (Down to the Microsecond)

Here's the most satisfying part of TPU profiling: **the math actually works.**

Imagine a single un-sharded Dense Matrix Multiply: `x @ W`.
- `x` = `(32, 1024, 8192)`
- `W` = `(8192, 32768)`

**The Math:**
- Total FLOPs = `2 × B × T × D × F`
- FLOPs = `2 × 32 × 1024 × 8192 × 32768` = **17.59 TFLOPs**.
- Peak hardware speed (TPU v5e) = **197 TFLOPs/s**.
- $\frac{17.59 \text{ TFLOPs}}{197 \text{ TFLOPs/s}} = \textbf{89.3 ms}$.

When looking at the actual JAX profile for this operation:
**Observed Execution Time = 94.1 ms!**

We hit nearly 95% MFU (Model FLOPs Utilization) on a single device, directly proving that our analytical models map perfectly to silicon.

---

## 🔀 Distributed Profiling: A Mock Transformer

When taking a simple Transformer and scaling it to an 8-core v5e topology, XLA shines—but it can also make terrible decisions if you don't constrain it.

Let's profile the `jnp.dot(x, w1)` projection:
- `x` = `[8, 1024, 8192]`
- `w1` = `[8192, 16384]` (FFW expansion)

**Step 1: Data Parallelism**
- Total FLOPs over 8 chips = 2.19 TFLOPs.
- Divided by 8 chips = **275 GFLOPs per chip**.
- Expected Time: `275e9 / 1.97e14` = **1.39 ms**.
- **Trace observed: 1.43 ms.** ✓

**Step 2: Logits Projection**
- Matrix is 2x larger (`d_ff` vs `vocab` where vocab is 32,768).
- Expected Time: `1.39 ms × 2` = **2.78 ms**.
- **Trace observed: 2.91 ms.** ✓

**Step 3: What about Fully Sharded Data Parallel (FSDP)?**
If we strictly constrain the compiler to execute FSDP on this small model (`jax.device_put(P('data', None))`), **performance craters.**
The profiler reveals `6 + 6 ms` wasted purely on inter-chip communication overhead (the All-Gather step). The computation becomes utterly **ICI (Inter-Chip Interconnect) Bound**.

---

## 🧠 Takeaways

1. **Traces Beat Guesses:** Stop wondering if a layer is compute bound (`T_math`) or memory bound (`T_hbm` / `T_ici`). The Perfetto trace explicitly labels nodes with execution times, revealing exactly who the culprit is.
2. **Compilers Make Mistakes:** XLA doesn't natively "know" the ideal multi-host topology unless you restrict it with `jax.lax.with_sharding_constraint`. Unchecked auto-sharding led to execution durations 5x to 8x worse than our hand-optimized data-parallel bounds!
3. **The `2BTDF / Hardware` Limit is Real:** When properly optimized, your models run exactly as fast as the fundamental physics equations dictate.

---

*Up Next: Part 10 - JAX Parallelism Deep Dive.*
