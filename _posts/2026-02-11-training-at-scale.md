---
layout: post
title: "Training at Scale: When Communication Becomes the Enemy"
date: 2026-02-11 01:00:00 +0900
description: Data Parallelism, FSDP, Tensor Parallelism — how to train 13B models without going communication-bound.
tags: jax tpu parallelism fsdp tensor-parallelism training mesh
categories: engineering
giscus_comments: false
related_posts: true
---

After mastering [Transformer math]({% post_url 2026-02-08-transformer-math %}), I tackled **Part 5: Training at Scale** from the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/training/).

This section is all about **parallelism strategies** — how to split a model across thousands of chips without becoming communication-bound.

---

## 🎯 The Big Idea

**Strong scaling** means adding chips should linearly increase throughput. But there's a catch:

> Each parallelism strategy has a **critical batch size** below which you're just paying for idle chips waiting on AllGathers.

$$
B/N > \text{critical\_threshold}
$$

---

## 🔧 The 4 Parallelism Strategies

| Strategy                 | What gets sharded                 | Critical batch size (TPU v5p) |
| ------------------------ | --------------------------------- | ----------------------------- |
| **Data Parallel (DP)**   | Activations (batch)               | `B/N > 2550`                  |
| **FSDP (ZeRO-3)**        | Activations + weights + optimizer | `B/N > 850` (3-axis)          |
| **Tensor Parallel (TP)** | Activations (D), weights (F)      | `Y < M_Y F/2550` (8-16 way)   |
| **FSDP + TP**            | Both!                             | `B/N > 235` (for LLaMA-2 13B) |

---

## 📝 My Solutions to Key Problems

### Problem 2: Memory Breakdown

**Question**: Training LLaMA-2 13B with BS=16M tokens using Adam (bf16 params, fp32 optimizer, checkpoint 3 matmuls/layer). How much memory?

**My Answer**:

- **Parameters** (bf16): `13B × 2 bytes = 26 GB`
- **Optimizer** (Adam, fp32): `13B × 8 bytes = 104 GB`
- **Activations** (bf16, checkpointing):
  - Per layer: `2(BF + BF + BD)` bytes
  - Total: `2 × 40 × 16M × (13.8K + 13.8K + 5.1K) = 42 TB`

**Key Insight**: Activations are **1,615× larger** than parameters! Without checkpointing, this would be ~280 TB. Gradient checkpointing isn't optional — it's survival.

---

### Problem 3: Sharding Strategy Selection

**Question**: Train with BS=3M tokens, 32K context on TPU v5p 16×16×16 (4096 chips). Which strategies work?

#### Pure Data Parallelism?

❌ **No** (two reasons)

- **Memory**: `130 GB` (params + optimizer) > `96 GB/chip`
- **Comms**: `B/N = 732 < 2550` → Communication-bound

#### Pure FSDP?

⚠️ **Barely, but comms-bound**

- **Memory**: `7.85 TB / 4096 = 2 GB/chip` ✓ Fits!
- **Comms**: `732 < 850` (for 3-axis FSDP) → **Communication-bound**

#### Mixed FSDP + Tensor Parallelism?

✅ **Yes!**

- **Compute-bound check**: `B/N > α²/(2M_X M_Y F) = 235`
  - Actual: `732 > 235` ✓
- **Optimal X**: `X_opt = √(BN M_X/(M_Y F)) = 1333`
  - Choose `X=1024` (FSDP), `Y=4` (TP)
- **Memory**: ~2 GB/chip ✓
- **Training time** (40% MFU): `6×13B×3M / (4.6e14×4096×0.4) = 300 ms/step`

**Key Insight**: Mixed FSDP+TP lets us **drop batch size to 235 tokens/chip** vs. 850 for pure FSDP — a **3.6× improvement**!

---

## 🧠 What I Learned

### The Mesh is Magic

JAX's `Mesh({X:16, Y:16, Z:16})` is **topology-agnostic**. You specify **logical** axes and JAX handles the physical routing. The same sharding code works on 4×4×4, 8×8×8, or 16×16×16 slices — JAX abstracts the hardware! 🤯

### Activations Dominate Memory

`42 TB` activations vs `26 GB` params = **1,615× difference**. Gradient checkpointing is mandatory, not optional.

### Each Strategy Has a Wall

Below the critical batch size, you're paying for chips that are idle waiting on communication.

### 32K+ Context Changes Everything

The book ignores long context because for training at T<32K, **MLP still dominates FLOPs** (~75%). But beyond 32K:

- **Attention becomes `O(T²)`** → Flash Attention is mandatory
- **Ring Attention** splits attention across chips
- **Context Parallelism** treats sequence dimension like batch

At T=128K, attention activations are **16× MLP activations**. This changes the entire game.

---

## 📐 Quick Reference

**When to use each strategy**:

- **DP**: Model fits on 1 chip, large batch
- **FSDP**: Model doesn't fit, medium batch (`B/N > 850`)
- **FSDP + TP**: Small batch (`B/N > 235` for LLaMA-2)
- **Pipeline**: Cross-pod scaling, GPU clusters

**Critical formulas**:

- **Optimal FSDP**: `X_opt = √(BN M_X/(M_Y F))`
- **Training time**: `6PT / (C × N × MFU)` (the **6PT rule**!)

---

## 📂 Code & Notes

All my notes and implementations: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

**Original Q&A**: [Google Doc](https://docs.google.com/document/d/150mR_Q3ju3ujFz752uEocnvLjjQUg0IaNOMmmPwqLRU/edit?usp=sharing)

**Next up**: Part 6 - Inference at Scale

---

_Sharding is economics: minimize communication, maximize compute._
