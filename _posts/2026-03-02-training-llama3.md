---
layout: post
title: "Training LLaMA 3 on TPUs: Putting Theory Into Practice"
date: 2026-03-02 20:00:00 +0900
description: Applying parallelism equations to real models — parameter counting, FLOPs, memory, and sharding for LLaMA 3-70B and 405B on TPU v5p pods.
tags: jax tpu llama training fsdp tensor-parallelism
categories: engineering
giscus_comments: false
related_posts: true
---

After building up the theory in [Part 5 (Training at Scale)]({% post_url 2026-02-11-training-at-scale %}), Part 6 is all about **doing the numbers on a real model**: LLaMA 3-70B on TPU v5p.

The key learning here: **these equations aren't just theory — they make real engineering decisions.**

---

## 📊 LLaMA 3-70B Architecture

| Param | Value |
|---|---|
| **L** (layers) | 80 |
| **D** (d_model) | 8,192 |
| **F** (d_ff) | 28,672 |
| **N** (n_heads) | 64 |
| **K** (n_kv_heads, GQA) | 8 |
| **H** (d_qkv) | 128 |
| **V** (vocab) | 128,256 |

---

## ✏️ Parameter Counting From Scratch

Working through the math myself:

| Component | Formula | Count |
|---|---|---|
| **Vocabulary** | `2 × D × V` | 2.1B |
| **Attention** | `2DH(N+K) × L` | 12.1B |
| **MLP** (SwiGLU) | `3 × D × F × L` | 56.3B |
| **Total** | | **~70.5B** ✓ |

The attention formula `2DH(N+K)` captures:
- Q projection + output: `2 × D × N × H` 
- K + V projections: `2 × D × K × H`
- Factor 2 covers GQA's separate K/V heads

**Key takeaway**: Without even knowing the architecture, you can re-derive parameter counts from 4 numbers (D, F, N, K).

---

## ⚡ FLOPs Estimates

Using the **6PT rule** (6 × Parameters × Tokens):

| Question | Answer |
|---|---|
| FLOPs per token | `6 × 70B = 420 GFLOPs` |
| Total (15T tokens) | `6 × 70B × 15T = 6.3e24 FLOPs` |
| Time on 1 v5p pod (8,960 chips, 40% MFU) | `6.3e24 / (8960 × 4.59e14 × 0.4) = 44 days` |

A single TPU would take **435 years**. A full pod: 44 days. That's what 8,960 chips buys you.

---

## 🧠 Memory Calculation

Training with BS=4M tokens, bf16 params, fp32 Adam optimizer, 4 checkpoints/layer:

| Component | Memory |
|---|---|
| Parameters (bf16) | `70B × 2 = 140 GB` |
| Optimizer (fp32 Adam) | `70B × 8 = 560 GB` |
| Activations (4 per layer) | **~110 TB** |

**The trap I fell into**: I initially estimated activations using `BD` shapes (~21 TB), but the FFW block outputs `BF` tensors — and since `F >> D` (28,672 >> 8,192), activations are 3.5× larger!

```
4 checkpoints × BF (dominant) × 80 layers × 2 bytes
= 4 × 4M × 28,672 × 80 × 2 = ~58 TB (just one checkpoint type)
Total ≈ 110 TB
```

**Minimum chips**: `111 TB / 96 GB = 1,156 chips` (not 230 as I first calculated!)

With 8,960 chips: `111 TB / 8960 ≈ 12.4 GB/chip` — well within the 96 GB limit.

---

## 🔀 Sharding Strategy

**Setup**: 4M token batch, 8,960 chips.

| Strategy | Feasible? | Reason |
|---|---|---|
| **Pure FSDP** (no seq parallelism) | ❌ | Only 1,024 sequences → 7,936 chips idle |
| **FSDP + sequence parallelism** | ❌ | `B/N = 446 < 850` → comms-bound |
| **Mixed FSDP + TP** | ✅ | `446 > 113 = α²/2F` → compute-bound! |

**Optimal X** (FSDP chips):
$$X_{opt} = \sqrt{\frac{B \cdot M_X \cdot N}{F \cdot M_Y}} = \sqrt{\frac{4.19M \times 2 \times 8960}{28,672}} = 1,618$$

→ Choose **X=2,048** (FSDP), **Y=4** (TP)

Final scheme: `1,024-way batch × 2-way sequence × 4-way tensor parallelism`

---

## 🏆 Worked Problems

### Q1: Scaling to 4 Pods (35,840 chips)

**Same 4M token batch, now split across 4 pods:**

1. **Cross-pod**: Pure data parallelism, 1M tokens/pod
   - DCN bound: `1M >> 71,360` → compute-bound ✓
2. **Per-pod**: Mixed FSDP+TP (same analysis, `B/N = 112 > 113`... barely!)
3. **Training time**: `6.3e24 / (4 × 8960 × 4.59e14 × 0.4) = 11 days`

**Linear scaling**: 1 pod → 4 pods = 44 days → 11 days. Perfect!

### Q2: LLaMA 3-405B on 8 Pods

**Architecture** (405B):

| Param | Value |
|---|---|
| L | 126 |
| D | 16,384 |
| F | 53,248 |
| N | 128 |
| K | 8 |

**Parameters**: `4.2B (vocab) + 71.4B (attention) + 330B (MLP) ≈ 406B` ✓

**FLOPs**: 
- Per token: `6 × 405B = 2.43 TFLOPs`
- 15T tokens: `3.65e25 FLOPs`

**Sharding on 8 pods** (71,680 chips, 500K tokens/pod):

| Check | Result |
|---|---|
| DCN bound: `B/pod > 71,360` | `500K >> 71K` ✓ |
| Pure FSDP: `B > 8960 × 850` | `500K < 7.6M` ❌ |
| Mixed FSDP+TP: `B/N > α²/2F = 61` | `B/N = 56 < 61` ❌ (barely!) |
| 16-way TP: `Y < F/2550 = 20` | `16 < 20` ✓ |

405B is **tight**. We're right at the communication-bound boundary for mixed FSDP+TP per pod. The practical answer: use 16-way TP with some accepted communication overhead.

**Training time**: `3.65e25 / (8 × 8960 × 4.59e14 × 0.4) ≈ 27.8 days`

---

## 🧠 What I Learned

**The core skill is plugging numbers into the 3-step decision framework:**

1. **Does it fit?** → Check `total memory / 96 GB` for minimum chips
2. **Are we compute-bound?** → Check `B/N > α²/2F` for FSDP+TP
3. **What's optimal?** → Use `X_opt = √(BN M_X / (F M_Y))`

**Key mistake to avoid**: When calculating activation memory, use the **largest intermediate shape** (`BF`, not `BD`). In LLaMA 3-70B, `F = 28,672` creates tensors 3.5× larger than the model dimension. This difference between 21 TB and 110 TB is the difference between needing 230 chips and 1,156 chips.

**The bigger realization**: These calculations tell you *before you start training* whether your setup will be efficient. In industry, getting this right means the difference between a training run that completes in 11 days vs. one that stalls at 50% GPU utilization.

---

## 📂 Code & Notes

Full notes: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

**Q&A Doc**: [Google Doc](https://docs.google.com/document/d/11DZQ8raX94P4AZ2b3f3GpPLzK5fI4-nwV1d9g0TAEvE/edit?usp=sharing)

**Chapter**: [JAX Scaling Book - Applied Training](https://jax-ml.github.io/scaling-book/applied-training/)

---

*Numbers don't lie. Run the equations before you run the training job.*
