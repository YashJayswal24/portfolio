---
layout: post
title: "Serving LLaMA 3-70B: From Theory to Production Numbers"
date: 2026-03-08 23:00:00 +0900
description: Real hardware serving decisions — KV cache sizing, topology selection, critical batch sizes, and the disaggregated serving ratio for LLaMA 3-70B on TPU v5e.
tags: jax tpu inference llama serving kv-cache latency throughput quantization
categories: engineering
giscus_comments: false
related_posts: true
---

After building up inference theory in [Part 7]({% post_url 2026-03-07-inference-at-scale %}), Part 8 applies it to an actual production question: **how do you actually serve LLaMA 3-70B on TPUs?**

This chapter is about turning equations into decisions.

---

## 🎯 The Hardware Choice: Why TPU v5e?

| TPU     | bf16 FLOPs/s | Cost/hr  | **FLOPs/$**  |
| ------- | ------------ | -------- | ------------ |
| H100    | 9.9e14       | $10.8    | 3.3e17       |
| v5p     | 4.59e14      | $4.2     | 3.9e17       |
| **v5e** | **1.97e14**  | **$1.2** | **5.8e17 ✓** |

Inference = optimize for **FLOPs per dollar**. v5e wins — it's cheaper even though each chip is slower.

---

## 📏 The Fundamental Constraint: KV Cache

For LLaMA 3-70B (K=8 KV heads, H=128, L=80 layers):
$$\text{KV bytes/token} = 2 \times K \times H \times L = 2 \times 8 \times 128 \times 80 = 160 \text{ kB}$$

At 32k context: **5.3 GB per sequence**. At BS=240: **1.3 TB of KV caches**.
This dwarfs the 70 GB of model parameters!

---

## 🔢 Working the Numbers: BS=32, 8k Context, int8

| Component        | Memory                        |
| ---------------- | ----------------------------- |
| Params (int8)    | 70 GB                         |
| KV caches (int8) | `160e3 × 8192 × 32 = 41.9 GB` |
| **Total**        | **~112 GB**                   |

Minimum: `112GB / 16GB/chip = 7 chips` → **4×2 (8 chips)**, 4×4 for headroom.

**Decode step latency** (B=32 < B_crit=120, so bandwidth-bound):

```
T = (70e9 + 41.9e9) / (8 chips × 8.1e11) = 17ms
Throughput = 32 / 0.017 / 8 = 235 tok/sec/chip
```

On 4×4: same throughput/chip, latency halves to **8.5ms**.

---

## ⚡ Critical Batch Sizes by Precision

| Mode                         | B_crit  | Reason                       |
| ---------------------------- | ------- | ---------------------------- |
| bf16 params + bf16 FLOPs     | **240** | `C/W_hbm = 1.97e14/8.1e11`   |
| **int8 params + bf16 FLOPs** | **120** | Half the bytes per param     |
| int8 params + int8 FLOPs     | **240** | FLOPs/s doubles, cancels out |

The "sweet spot" is **int8 params + bf16 FLOPs**: half as many chips needed, minimal quality degradation, and B_crit=120 is way more achievable than 240.

---

## 📊 Minimum Topology by Precision

| Precision | Param bytes | Min topology       |
| --------- | ----------- | ------------------ |
| bf16      | 140 GB      | **4×4 (16 chips)** |
| int8      | 70 GB       | **4×2 (8 chips)**  |
| int4      | 35 GB       | **2×2 (4 chips)**  |

A surprising result: when you fill HBM completely at each topology, **all three give the same max batch (~44 seqs) and same step latency (~20ms)**:

```
T = HBM_per_chip / W_hbm = 16 GB / 8.1e11 = 19.8ms
```

The topology choice changes cost and chip count, not necessarily latency.

---

## 📈 QPS per Chip (512 median decode tokens)

```
QPS/chip = batch / (step_time × decode_steps × N_chips)
```

| Precision | N chips | Max batch | QPS/chip |
| --------- | ------- | --------- | -------- |
| bf16      | 16      | 44        | **0.27** |
| int8      | 8       | 44        | **0.54** |
| int4      | 4       | 44        | **1.07** |

**int4 gives 4× better QPS/chip than bf16** — but validate accuracy before deploying!

---

## 🚀 Doubling Topology (bf16: 4×4 → 4×8)

More chips = more HBM for KV = bigger batches:

```
Available for KV: 512GB - 140GB = 372GB
Max batch: 372 / 2.62 GB/seq = 143 sequences (was 44)
```

**3.3× more total throughput, 1.6× more per-chip** — super-linear gains from scaling!

But can we actually use a 4×8 with model parallelism?

---

## 🔀 Sharding on 4×8: ICI Bounds

For generation, only model parallelism works. Two bottlenecks to check:

**FLOPs-ICI bound**: `Y > 2F/α_ICI = 2 × 28,672 / 2,189 = 26`

**HBM-ICI bound** (at B=143): `Y > F/(B×β) = 28,672/(143×9) ≈ 22`

Both give ~22–26 max chips before ICI bottleneck. We have 32 on a 4×8 → **ICI-bound at large batch!**

✅ Serve bf16 on **4×4 max** for model parallel  
✅ For int8/int4 (smaller KV → smaller batch → higher Y_max), 4×8 may work

---

## ⏱️ Prefill is Slow

At 40% MFU on 16 v5e chips:
$$T_{prefill} = \frac{2 \times 70\text{B params} \times 8192 \text{ tokens}}{16 \times 1.97e14 \times 0.4} = \textbf{0.91 seconds!}$$

A single long prompt takes nearly a second. This means prefill can be the **bottleneck in production**, not generation!

---

## 🏗️ Disaggregated Serving: The 3:1 Ratio

**Setup**: 8192 prefill, 512 decode, bf16, 16 chips, BS=32

To keep both prefill and generate servers fully utilized:

$$\frac{P_{\text{servers}}}{0.91 \text{s}} = \frac{32 \times G_{\text{servers}}}{0.017\text{s} \times 512 \text{steps}}$$

$$\boxed{P = 3G}$$

**You need 3× as many prefill servers as generate servers!**

This is why real production stacks (like Google's serving infrastructure) physically separate prefill and generation onto different server pools that can scale independently.

---

## 🧠 The Serving Decision Framework

```
1. Precision?         → B_crit tells you (120 for int8+bf16, 240 otherwise)
2. Min topology?      → ceil(param_bytes / 16 GB/chip)
3. Max batch?         → (N × 16GB - params) / KV_per_seq
4. Step latency?      → 16GB / W_hbm ≈ 20ms when HBM full
5. ICI-safe?          → Y < min(2F/2200, F/(β×B)) ≈ 22–26 for 70B
6. Prefill:generate?  → P/G = prefill_time / (B / decode_rate)
```

---

## 📂 Notes & References

Full solutions: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

**Q&A Doc**: [Google Doc](https://docs.google.com/document/d/1ZNcs3f8CnHKmdMIh9siDrp8uO57CnV0sJjiHKgQ7w78/edit?usp=sharing)

**Chapter**: [JAX Scaling Book - Applied Inference](https://jax-ml.github.io/scaling-book/applied-inference/)

---

_Math decides your topology. KV cache decides your batch size. Both decide your bill._
