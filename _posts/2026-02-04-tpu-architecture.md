---
layout: post
title: "TPU Architecture: Understanding the Bandwidth Hierarchy"
date: 2026-02-04 23:30:00 +0900
description: From MXUs and VMEM to ICI and DCN — mapping the performance landscape of Google's TPUs.
tags: jax tpu systems-engineering hardware bandwidth
categories: engineering
giscus_comments: false
related_posts: true
---

After mastering [roofline analysis]({% post_url 2026-02-03-roofline-analysis %}), I dove into **Part 2: TPU Architecture** from the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/tpus/).

This section is all about understanding the **bandwidth hierarchy** that governs TPU performance.

---

## 🏗️ The TPU Stack

A TPU is fundamentally simple:

1. **MXU** (Matrix Multiply Unit) — Systolic array doing `bf16[8,128] @ bf16[128,128]` every 8 cycles
2. **VMEM** (Vector Memory) — 128 MiB on-chip scratchpad
3. **HBM** (High-Bandwidth Memory) — 16-96 GB depending on generation
4. **ICI** (Inter-Chip Interconnect) — Nearest-neighbor torus topology
5. **DCN** (Datacenter Network) — Host-to-host networking

**The Golden Rule**: Communication speed decreases ~10x at each level.

---

## 📊 The Bandwidth Hierarchy

| Link           | Bandwidth            | Use Case                    |
| -------------- | -------------------- | --------------------------- |
| **HBM ↔ VMEM** | 1-3 TB/s             | Loading weights/activations |
| **ICI**        | 90-180 GB/s per link | Cross-chip sharding         |
| **PCIe**       | 16-32 GB/s           | Host ↔ TPU                  |
| **DCN**        | 6-12 GB/s            | Cross-pod communication     |

---

## 📝 My Solutions to the Problems

### Problem 1: Bounding LLM Latency

**Question**: How long to load a 200B parameter model (bf16) across 32 TPU v4p?

**My Answer**:

- **Size**: `200e9 × 2 bytes = 4e11 bytes`
- **Total HBM BW**: `32 × 1.2e12 = 3.84e13 bytes/s`
- **Time**: `4e11 / 3.84e13 = **10 ms**`

**Insight**: Even massive models load in milliseconds when parallelized across enough chips.

---

### Problem 2: TPU Pod Specs

**Question**: Calculate total FLOPs, memory, and hosts for full v5e and v5p pods.

**My Answers**:

**TPU v5e** (16×16 = 256 chips):

- **FLOPs/s**: `5.04e16` (50 PetaFLOPs)
- **HBM**: `4 TB`
- **Hosts**: `8`

**TPU v5p** (16×20×28 = 8960 chips):

- **FLOPs/s**: `4.11e18` (4 ExaFLOPs!)
- **HBM**: `860 TB`
- **Hosts**: `2240`

**Insight**: A single v5p pod is one of the **most powerful supercomputers in the world**. And Google has many.

---

### Problem 3: PCIe Operational Intensity

**Question**: What batch size needed to saturate a TPU v6e over PCIe?

**Setup**: `bf16[B,D] @ bf16[D,4D]` with PCIe BW = 1.5e10 bytes/s

**My Answer**:

- **Critical AI**: `9.20e14 / 1.5e10 = 6.13e4`
- **Threshold**: `B > 6133 tokens`

**Insight**: PCIe is **~100x slower than HBM**. If your weights are on the host, you need **massive batches** to avoid starvation.

---

### Problem 4: General Matmul Latency

**Question**: Time for `int8[16K,4K] @ int8[B,4K]` on TPU v5e?

**My Answer**:

- **Compute-bound threshold**: `B > 250`
- **Time** (B > 250): `T = B / 2.93e6` seconds

**From VMEM**: If VMEM BW is ~20x HBM → threshold drops to `B > 12`

**Insight**: VMEM's higher bandwidth lets you be compute-bound at **20x smaller batches**.

---

### Problem 5: ICI Bandwidth & Torus Topology

**Question**: Send `bf16[8,128,8K]` from TPU{0,0} → TPU{3,3} in a 4×4 slice.

**My Answer**:

1. **First byte**: 6 hops × 1μs = **6 μs**
2. **Full transfer**:
   - Can split across **2 disjoint paths** (torus!)
   - Time: `1.7e7 / (2 × 4.5e10) = **189 μs**`

**Insight**: The **torus topology** effectively doubles bandwidth by enabling multi-path routing.

---

### Problem 6: The Full Pipeline (Hard)

**Question**: Copy sharded `int8[128K,128K]` from host DRAM to one TPU, multiply by `bf16[8,128K]`.

**My Breakdown**:

1. **DCN transfer** (host → chips): 66 ms
2. **ICI aggregation** (chips → TPU{0,0}): **167 ms** ⚠️
3. **HBM → MXU**: 19 ms
4. **Compute**: 1.3 ms

**Bottleneck**: **ICI** at 167 ms — even for a "small" 15 GB transfer.

**Insight**: When aggregating data across many chips, **ICI bandwidth dominates** — orders of magnitude more than compute time.

---

## 🧠 What I Learned

1. **VMEM is underrated**: 20x higher bandwidth means 20x smaller batch size thresholds.
2. **ICI is the real bottleneck**: Even "fast" nearest-neighbor links become slow when aggregating across 16+ chips.
3. **Topology matters**: The torus structure isn't just elegant — it **doubles your effective bandwidth**.

At Samsung, when we shard models across TPUs, we're not just dividing work — we're **navigating a complex hierarchy of bandwidths**. The difference between 10ms and 200ms latency often comes down to **which link the data crosses**.

---

## 📂 Code & Notes

All my notes and implementations: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

**Original Q&A**: [Google Doc](https://docs.google.com/document/d/1q3feWPBbFnTLwdJJ7PkSWaHh-fRHcii6JmBY9bpol6s/edit?usp=sharing)

**Next up**: Part 3 - Sharding Strategies

---

_Understanding the hierarchy is half the battle._
