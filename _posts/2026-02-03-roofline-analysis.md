---
layout: post
title: "Roofline Analysis: When Does Your Model Hit the Wall?"
date: 2026-02-03 23:15:00 +0900
description: Understanding compute-bound vs. memory-bound operations, arithmetic intensity, and how quantization changes the game.
tags: jax tpu systems-engineering llm roofline quantization
categories: engineering
giscus_comments: false
related_posts: true
---

After understanding the *why* behind strong scaling in [Part 0]({% post_url 2026-02-02-scaling-llms-intro %}), I dove into **Part 1: Roofline Analysis** from the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/roofline/).

This section is all about answering one question: **Is my operation limited by compute, memory, or communication?**

---

## 🎯 The Roofline Model

Every operation has an **Arithmetic Intensity (AI)**:

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Loaded}}
$$

When AI exceeds the hardware's threshold, we're **compute-bound** (good). Below that, we're **memory-bound** (wasting FLOPs/s).

For TPU v5e:
- Peak FLOPs/s: `1.97e14`
- HBM Bandwidth: `8.2e11 bytes/s`
- **Critical AI**: `~240 FLOPs/byte`

---

## 📝 My Solutions to the Problems

### Problem 1: int8 Matmul

**Question**: How does int8 quantization change the roofline?

**My Analysis**:
- **Bytes**: `BD + DF` (loaded), `BF` (written)
- **OPs**: `2*BDF` (int8 has 2x peak OPs/s vs. bf16)
- **AI**: `2*BDF / (BD + DF + BF) ≈ 2*B` (when `B << D, F`)
- **Threshold**: `B > 243` (vs. 240 for bf16)

**Insight**: int8 quantization barely changes the batch size threshold — you still need ~240 tokens to be compute-bound.

---

### Problem 2: Mixed Precision (int8 Weights + bf16 Activations)

**Question**: What if we quantize weights but keep activations in bf16?

**Setup**: `bf16[B,D] * int8[D,F] → bf16[B,F]`

**My Analysis**:
- **FLOPs**: Still `2*BDF` (bf16 compute)
- **Bytes**: `2*BD + DF + 2*BF ≈ DF` (weights are 1 byte each!)
- **AI**: `2*BDF / DF = 2*B`
- **Threshold**: `B > 120`

**Insight**: This is the **real win** of quantization. By halving the weight size, we become compute-bound at **half the batch size** (120 vs. 240).

---

### Problem 3: Roofline Visualization

**Question**: Plot peak FLOPs/s vs. batch size for different matrix dimensions.

**My Solution**: Created an interactive [Desmos plot](https://www.desmos.com/calculator/hrgrfz5hkt?embed) showing the roofline transition.

<iframe src="https://www.desmos.com/calculator/hrgrfz5hkt?embed" width="100%" height="400" style="border: 1px solid #ccc" frameborder="0"></iframe>

**Insight**: Larger matrices (`D=F=4096`) reach peak FLOPs/s at smaller batch sizes than smaller matrices (`D=F=1024`).

---

### Problem 4: Batched Matmul

**Question**: What if we have a **different matrix per batch element**?

**Setup**: `int8[B,D] * int8[B,D,F] → int8[B,F]`

**My Analysis**:
- **Bytes**: `BD + BDF + BF`
- **OPs**: `2*BDF`
- **AI**: `2*DF / (D + DF + F) ≈ 2` (when `DF >> D, F`)

**Insight**: The AI is **fixed at ~2**, regardless of batch size. This operation will almost always be **memory-bound**.

---

### Problem 5: GPU Rooflines (NVIDIA H100)

**Question**: What's the batch size threshold for an H100 GPU?

**Specs**:
- Peak FLOPs/s: `9.895e14` (accounting for structured sparsity)
- HBM Bandwidth: `3.35 TB/s`

**My Answer**:
- **Critical AI**: `9.895e14 / (2 * 3.35e12) ≈ 295`
- **Threshold**: `B > 295`

**Insight**: GPUs have a slightly higher compute-to-bandwidth ratio than TPUs, requiring **larger batch sizes** to saturate compute (~295 vs. ~240).

---

## 🧠 What I Learned

1. **Quantization changes the game**: int8 weights let you be compute-bound at **half** the batch size.
2. **Batched ops are tricky**: Operations with different matrices per sample have **fixed AI** and are almost always memory-bound.
3. **Hardware matters**: TPU vs. GPU rooflines differ by ~20%, making batch size tuning hardware-specific.

At Samsung, this means when we quantize models, we're not just saving memory — we're **shifting the entire performance profile**.

---

## 📂 Code & Notes

All my notes and future implementations are in the [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax) repository.

**Next up**: Part 2 - TPU Architecture

---

*Understanding when you hit the wall is the first step to breaking through it.*
