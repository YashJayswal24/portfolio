---
layout: post
title: "GPUs for LLMs: The Same Rooflines, Different Numbers"
date: 2026-03-17 23:00:00 +0900
description: How GPUs differ from TPUs at the chip level, NVLink networking, collective communications, and why pipeline parallelism is the key to training LLaMA-3 70B at scale.
tags: gpu nvidia h100 nvlink llm scaling parallelism roofline
categories: engineering
giscus_comments: false
related_posts: true
---

Everything from the TPU chapters applies to GPUs too. The same roofline equations, the same communication bottlenecks, the same critical batch sizes. The _numbers_ just change.

---

## 🏗️ GPU Architecture: CUDA Cores vs Tensor Cores

A modern GPU has two radically different types of compute:

- **CUDA cores (vector ALUs)**: One multiply per clock. An H100 has `132 SMs × 4 subpartitions × 32 cores = 16,896` CUDA cores running at 1.59–1.98 GHz → **27–33.5 fp32 TFLOPs/s**.

- **Tensor Cores**: Run as matrix engines inside each SM. Can do a 4×4 matrix multiply per clock → **990 bf16 TFLOPs/s** total.

Tensor Cores give you **30× more FLOPs/s** than CUDA cores. The takeaway: CUDA cores are essentially overhead. Everything performance-critical happens on Tensor Cores via large matmuls.

**Comparison to TPU v5p:**

- H100: 16,896 vector ALUs
- TPU v5p: 8,192 vector ALUs (VPU lanes)
- Factor of 2× more GPUs vector lanes — but both clock at ~same frequency, so practical difference is smaller than it first appears.

---

## 📏 Critical Batch Size: Still ~295 Tokens

Just like TPUs (~240 tokens), an H100 becomes **compute-bound at ~295 tokens** per chip in a matmul:

$$B_\text{crit} = \frac{\text{FLOPs/s}}{\text{HBM BW}} = \frac{990 \text{ TFLOPs/s}}{3.4 \text{ TB/s}} \approx 295$$

For a B200, this stays almost identical at 281 — because both FLOPs and bandwidth scaled up together (2.25× FLOPs, 2.35× bandwidth). The hardware balance point barely moves generation-to-generation.

---

## 🔌 NVLink and NVSwitch: TPU ICI Analogs

On TPUs, inter-chip interconnect (ICI) had a fixed bandwidth with a defined ring topology. On GPUs:

- **NVLink**: Direct GPU-to-GPU links (18 links × 25 GB/s = 450 GB/s full-duplex per GPU on H100).
- **NVSwitch**: An in-node all-to-all switch that gives every GPU access to every other GPU's bandwidth simultaneously.

This creates a **3.6 TB/s total intra-node bandwidth** for 8×H100 nodes — a **full-mesh** fabric where bisection bandwidth = total bandwidth.

**Key collective costs on 8×H100:**

| Operation            | Array                       | Time                            |
| -------------------- | --------------------------- | ------------------------------- |
| AllGather            | 536 MB                      | `7/8 × 536e6 / 450e9 = 1.04 ms` |
| AllGather (pop quiz) | bf16[1024, 16384] = 33.6 MB | `65 µs`                         |

Beyond the node: **InfiniBand @ 400 GB/s** — same idea, but 10× lower bandwidth than intra-node NVLink.

---

## 🧮 Rooflines for LLM Training

Given H100 specs: **990 TFLOPs/s** compute, **450 GB/s** intra-node, **400 GB/s** cross-node:

| Strategy                            | Critical batch (per GPU)     | Why                       |
| ----------------------------------- | ---------------------------- | ------------------------- |
| **Tensor Parallelism** (intra-node) | `990e12/450e9 = 2200 tokens` | AllReduce over NVLink     |
| **Data Parallelism** (cross-node)   | `990e12/400e9 = 2475 tokens` | AllReduce over InfiniBand |
| **2-node DP**                       | `~1237 tokens`               | (N-1)/N factor for N=2    |

For B200 DGX SuperPods: FLOPs jumped 2.25× but scale-out BW stayed the same:
$$B_\text{crit, cross-node} = \frac{2250 \text{ TFLOPs}}{400 \text{ GB/s}} = 5625 \text{ tokens/GPU}$$

B200 is notably _harder_ to be compute-bound for cross-node data parallelism.

---

## 🦙 Sharding LLaMA-3 70B on 4096 H100s

**Memory budget:**

- Weights (bf16): 70B × 2 = 140 GB
- Adam state (fp32): 70B × 8 = 560 GB
- **Total: 700 GB → minimum 9 GPUs (2 nodes)**

**Training time:**
$$T = \frac{6 \times 70\text{B params} \times 15\text{T tokens}}{0.45 \times 4096 \times 990\text{ TFLOPs/s}} = 3.44 \times 10^6 \text{ s} \approx \mathbf{40 \text{ days}}$$

**Max model parallelism (intra-node TP):**
$$Y < \frac{F}{W_{\text{link}} / C} = \frac{28672}{450\text{GB}/990\text{TFLOPs}} = 14.4 \rightarrow \mathbf{8\text{-way TP}}$$

**With 8-way TP + 512-way DP (pure):**

- Memory check: `700 GB / 8 = 87.5 GB` → **Doesn't fit in 80 GB HBM!** ❌
- Batch per GPU: `4M / 4096 = 976 tokens` → below critical 2475 → **Not compute-bound!** ❌

**ZeRO-3 + 8-way TP:**

- Memory: ✓
- But ZeRO doubles AllReduce traffic → critical batch doubles to ~5000. At 976 tokens/GPU → **still not compute-bound!** ❌

**8-way Pipeline Parallelism:**
Each pipeline stage spans 8 nodes, aggregating their cross-node bandwidth:
$$\text{Effective BW} = 8 \times 400 \text{ GB/s} = 3200 \text{ GB/s}$$
$$B_\text{crit} = \frac{990\text{ TFLOPs}}{3200 \text{ GB/s}} = 309 \text{ tokens/GPU}$$

With 976 tokens/GPU: `976 > 309` → ✅ **Compute-bound with pipeline + TP!**

**The winner: 8-way TP (intra-node) × 8-way pipeline × 64-way DP = 4096 GPUs.**

---

## 📏 Key Takeaways

1. **Same physics, different constants.** GPU rooflines use the same equations as TPUs. The H100 critical batch (~295) is startlingly close to the TPU v5e critical batch (~240).

2. **GPU SMEM is half of TPU VMEM.** H100 has 66MB SMEM+registers vs ~120MB TPU VMEM. On TPUs, VMEM spill is cheap. On GPUs, L2 spills can cost you.

3. **NVSwitch = full mesh.** Unlike TPU ring topologies, NVSwitch gives bisection bandwidth = total node bandwidth. AllGathers don't degrade with parallelism degree (within a node).

4. **Pipeline parallelism aggregates inter-node bandwidth.** It's the key that makes large-scale H100 training viable with the relatively slow 400 GB/s InfiniBand links.

5. **B200 doesn't solve the cross-node problem.** 2.25× FLOPs with the same 400 GB/s InfiniBand means cross-node critical batch grows from 2475 → 5625. You need GB200 NVL72 (72-GPU nodes) to fix this.

---

_Full solutions notebook: [Model_scaling_jax on GitHub](https://github.com/YashJayswal24/Model_scaling_jax/tree/main/12_gpus_for_llms)_
