---
layout: post
title: "Transformer Math: The 6PT Rule and Other Accounting Tricks"
date: 2026-02-08 00:15:00 +0900
description: Parameter counting, FLOPs estimation, attention complexity — the economics of modern LLMs.
tags: jax tpu transformers flops attention moe
categories: engineering
giscus_comments: false
related_posts: true
---

After mastering [sharding strategies]({% post_url 2026-02-07-sharding-strategies %}), I dove into **Part 4: Transformer Math** from the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/transformers/).

This section is all about **counting FLOPs and parameters** — the economics that govern LLM training.

---

## 🎯 The Big Idea

Training a Transformer is fundamentally simple:

$$
\text{FLOPs} \approx 6 \times \text{Parameters} \times \text{Tokens}
$$

This `6PT` rule holds for reasonable context lengths (`T < 8D`) and explains why training costs scale the way they do.

---

## 📊 Transformer Accounting

| Component     | Params/layer | FLOPs/layer (training) |
| ------------- | ------------ | ---------------------- |
| **MLP**       | `3DF`        | `18BTDF`               |
| **Attention** | `4D²`        | `24BTD² + 12BT²D`      |
| **Vocab**     | `DV`         | `12BTDV`               |

**Key Takeaway**: MLP dominates params (75%) and FLOPs (when `T < 8D`).

---

## 📝 My Solutions to Key Problems

### Problem 1: Parameter Counting

**Question**: `D=4096`, `F=16K`, `V=32K`, `L=64`. Total params? Attention fraction? KV cache/token?

**My Answer**:

- **MLP**: `3DF × 64 = 12.6B`
- **Attention**: `4D² × 64 = 4.3B`
- **Vocab**: `DV = 131M`
- **Total**: `~16B params`

**Attention fraction**: `4D²/(4D² + 3DF) = 1/4`

**KV cache**: `2LKH = 512 KB/token` (int8)

**Key Insight**: Even with multi-head attention, MLP **dominates** parameter count.

---

### Problem 4: Attention Complexity

**Question**: Arithmetic intensity of self-attention? When is it FLOPs-bound?

**My Answer**:

- **Bytes**: `3BTNH` (K, Q, V tensors)
- **FLOPs**: `4BT²NH` (Q·Kᵀ + Attention·V)
- **AI**: `4T/3` (grows linearly with `T`)
- **FLOPs-bound when**: `T > 240` (on TPU v5e)

**Key Insight**: Attention becomes compute-bound at moderate sequence lengths (~240 tokens). For full training runs, attention is rarely the bottleneck.

---

### Problem 5: When Does Attention Dominate?

**Question**: At what sequence length do attention FLOPs equal projection FLOPs?

**My Answer**:

- **Attention**: `2BT²NH`
- **Projections** (Q,K,V,O): `4BTDNH`
- **Equal when**: `T = 2D`

**Key Insight**: For `D=8192`, attention dominates beyond **16K tokens**. Below that, projections (matmuls) dominate.

---

### Problem 7: DeepSeek v3 Efficiency

**Question**: Trained for 2.79M H800-hours on 14.8T tokens with 37B activated params (FP8). What's the MFU?

**My Answer**:

- **Required FLOPs**: `6 × 37e9 × 14.8e12 = 3.28e24`
- **H800 FP8 FLOPs/s**: `1.513e15`
- **Available compute**: `1.513e15 × 2.79e6 × 3600 = 1.52e25`
- **Utilization**: `3.28e24 / 1.52e25 ≈ 21.5%`

**Key Insight**: **~20% MFU** is typical for MoE models due to communication overhead and expert routing.

---

### Problem 8: MoE Batch Size Requirements

**Question**: What batch size makes an MoE compute-bound on TPU v5e? For DeepSeek (`E=256`, `k=8`)?

**My Answer**:

- **Activates**: `k` experts per token
- **Loads**: All `E` experts
- **Compute-bound when**: `B > (E/2k) × 240`

**DeepSeek**: `B > (256/16) × 240 = 3,840 tokens`

**Key Insight**: MoE requires **16× larger batches** than dense models to saturate compute (because you load all `E` experts but only use `k`).

---

## 🧠 What I Learned

1. **The 6PT rule is real**: Training FLOPs are almost exactly `6 × params × tokens` for reasonable context.
2. **Attention isn't the bottleneck** until `T > 2D` — for most training, **MLPs dominate**.
3. **MoE's hidden cost**: You need `(E/k)×` the batch size to be compute-bound. DeepSeek needs **3,840 tokens/batch** vs. 240 for dense models.

At Samsung, when we estimate training costs, we're basically computing `6PT` and dividing by hardware FLOPs/s. Everything else is overhead.

---

## 📂 Code & Notes

All my notes and implementations: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

**Original Q&A**: [Google Doc](https://docs.google.com/document/d/1ByEHwNMeRgl9OI_fTwRn6jY4IeHdX6Q2kFKmTgZ4oQg/edit?usp=sharing)

**Next up**: Part 5 - Training at Scale

---

_Economics beats elegance._
