---
layout: post
title: "Sharding Strategies: The Art of Distributed Matrix Multiplication"
date: 2026-02-07 18:55:00 +0900
description: AllGather, ReduceScatter, AllToAll — mastering the 4 communication primitives that make LLM training scale.
tags: jax tpu systems-engineering sharding communication allgather
categories: engineering
giscus_comments: false
related_posts: true
---

After understanding [TPU architecture]({% post_url 2026-02-04-tpu-architecture %}), I tackled **Part 3: Sharded Matrices** from the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/sharding/).

This section is all about the **4 core communication primitives** that enable distributed matrix multiplication.

---

## 🎯 The Big Idea

When you split a matrix across 64 TPUs and multiply it, you need to:

1. **Move data** between chips (communication)
2. **Do local math** on each chip (computation)

The art is choosing **which data to move, when, and how**.

---

## 🔧 The 4 Core Primitives

| Primitive         | What it does         | Cost           |
| ----------------- | -------------------- | -------------- |
| **AllGather**     | `[Aₓ,B] → [A,B]`     | `V / W_ici`    |
| **ReduceScatter** | `[A,B]{Uₓ} → [Aₓ,B]` | `V / W_ici`    |
| **AllReduce**     | `[A,B]{Uₓ} → [A,B]`  | `2V / W_ici`   |
| **AllToAll**      | `[A,Bₓ] → [Aₓ,B]`    | `V / (4W_ici)` |

**The Golden Rule**: Cost depends only on **array size** and **bandwidth**, NOT number of devices!

---

## 📝 My Solutions to Key Problems

### Problem 4: Strategy Comparison

**Question**: Multiply `X[B,D] · Y[Dₓ,F]`. Which is faster?

1. AllGather Y, then multiply
2. Multiply locally, then AllReduce

**My Analysis**:

- **Strategy 1**: `T_comm = 2DF/W`, `T_comp = 2BDF/W_flop`
  - Compute-bound when `B > W_flop/W_ici`
  - **Best for large batches**
- **Strategy 2**: `T_comm = 4BF/W`, `T_comp = 2BDF/(X·W_flop)`
  - Compute-bound when `D > 2X·W_flop/W_ici`
  - **Best for large models**

**Key Insight**: Large batches favor Strategy 1. Large models favor Strategy 2.

---

### Problem 6: Common Sharding Patterns

**Question**: Analyze 3 real-world sharding schemes on 4×4 TPU v5e.

#### Pattern 1: `A[Iₓ,Jᵧ] · B[Jᵧ,K] → C[Iₓ,K]`

- **AllGatherᵧ** twice → `T_comm = 2JK/(Y·W_ici)`
- Compute: `2IJK/(XY·W_flop)`

#### Pattern 2: `A[Iₓ,J] · B[Jₓ,Kᵧ] → C[Iₓ,Kᵧ]` (Training)

- **AllGatherₓ** on J → `T_comm = JK/(Y·W_ici)`
- This is **Data Parallel + Tensor Parallel + ZeRO**

#### Pattern 3: `A[Iₓ,J] · B[J,Kᵧ] → C[Iₓ,Kᵧ]` (Inference)

- **No communication!**
- This is **pure Tensor Parallel + Data Parallel**

**Key Insight**: Inference can be **communication-free** with proper sharding.

---

### Problem 7: Transformer Block Memory Constraints

**Question**: Shard `In[128,8K] · Wᵢₙ[8K,32K] · Wₒᵤₜ[32K,8K]` on 2×2 v5e with 300MB/TPU.

**My Strategy**:

- **Problem**: Each weight matrix is 536 MB (too big!)
- **Solution**: Shard weights as `Wᵢₙ[D,Fₓᵧ]`, `Wₒᵤₜ[Fₓᵧ,D]`
- **Trick**: Fuse `Wᵢₙ · Wₒᵤₜ` first, then ReduceScatter
  1. `Wᵢₙ · Wₒᵤₜ → W[D,D]{Uₓᵧ}` (local multiply)
  2. `ReduceScatterₓᵧ → W[D,Dₓᵧ]`
  3. `AllGather + In · W → Out[B,Dₓᵧ]`

**Key Insight**: Intermediate fusion **halves communication** vs. sequential matmuls.

---

### Problem 10: Why is AllToAll 4× Faster?

**Question**: Explain the factor-of-4 speedup for AllToAll vs. AllGather in bidirectional rings.

**My Analysis**:

**Unidirectional**:

- **AllGather**: Every device sends full shard around the ring → `N²` scalars total
- **AllToAll**: Each device sends partial shards → `N²/2` scalars
- **Ratio**: `2×`

**Bidirectional**:

- **AllGather**: `D/2` hops, `2N²/D` bytes/hop → `N²` total
- **AllToAll**: `D/4` hops, `N²/D` bytes/hop → `N²/4` total
- **Ratio**: `4×`

**Key Insight**: Bidirectional AllToAll benefits from **halved hops** AND **halved bytes/hop**.

---

## 🧠 What I Learned

1. **Communication ≠ scaling bottleneck**: AllGather on 4 chips takes the same time as on 64 chips (bandwidth-bound).
2. **Strategy beats scale**: Choosing AllGather-then-multiply vs. multiply-then-AllReduce can **2× performance**.
3. **Inference is "free"**: With `A[Iₓ,J] · B[J,Kᵧ]`, there's **zero cross-chip communication**.

At Samsung, when we shard models, we're not just dividing work — we're **choosing a communication strategy** that determines whether we hit 10ms or 200ms latency.

---

## 📂 Code & Notes

All my notes and implementations: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

**Original Q&A**: [Google Doc](https://docs.google.com/document/d/16ogD2ipwt2AZetSoC5S4L7_hvWEAY00QrQBm6JExuqw/edit?usp=sharing)

**Next up**: Part 4 - Transformer Math

---

_Sharding isn't about splitting — it's about orchestrating._
