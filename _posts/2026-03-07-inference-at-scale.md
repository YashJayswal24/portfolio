---
layout: post
title: "Transformer Inference: Two Problems in Disguise"
date: 2026-03-07 18:00:00 +0900
description: Prefill is compute-bound like training. Generation is always memory-bandwidth-bound. The KV cache changes everything.
tags: jax tpu inference kv-cache moe latency throughput
categories: engineering
giscus_comments: false
related_posts: true
---

After [training LLaMA 3]({% post_url 2026-03-02-training-llama3 %}), Part 7 is about **serving it**. The fundamental insight: inference isn't one problem — it's two completely different problems wearing the same coat.

---

## 🎯 The Core Dichotomy

|          | **Prefill**                   | **Generation**                        |
| -------- | ----------------------------- | ------------------------------------- |
| What     | Process prompt, fill KV cache | Sample one token at a time            |
| Bound    | **Compute** (like training)   | **Memory-bandwidth (always!)**        |
| Sharding | TP + Sequence parallel        | **Model parallel only**               |
| Batch    | Can batch prompts together    | Sequential dependency — hard to batch |

The key new data structure: **the KV cache** = `2 × L × K × H × T` bytes per sequence.

For the 18.4B model I worked through (L=64, K=8, H=256): **262 kB per token!** At 128k context, one sequence uses 33.5 GB — nearly 2× the entire model.

---

## ⚡ Critical Batch Sizes

During generation, to be compute-bound on linear layers:
$$B > B_{crit} = \frac{C}{W_{hbm}} \times \frac{\text{bits/param}}{\text{bits/activation}}$$

| Setup                           | B_crit          |
| ------------------------------- | --------------- |
| bf16 params on v5e              | **240 tokens**  |
| int8 params + bf16 FLOPs on v5e | **120 tokens**  |
| MoE (E=16, k=2)                 | **1920 tokens** |

Attention is **always** bandwidth-bound during generation, regardless of batch size. You can never escape this.

---

## 🔑 The Pop Quiz Trap

**Problem**: Generate from a 30B model (int8) on 4×4 v5e, batch=4, 8192-context, 100kB/token KV cache.

```
T_step = (B × KV_size + params) / (N_chips × W_hbm)
= (4 × 8192 × 100e3 + 30e9) / (16 × 8.1e11)
= 2.5 ms
```

**My mistake**: I only counted params (`30e9/16/8.1e11 = 2.3ms`) and forgot the KV cache term. Even at batch=4, KV contributes ~3.3 GB. **Always include KV in your latency estimate!**

---

## 📊 Problems & Key Results

### Q2: KV Cache Dominates Memory

Max batch for 128k context on 4×4 v5e:

```
B × (262e3 × 128e3) + 18.4e9 ≤ 16 × 16e9
B × 33.5e9 ≤ 237.6e9
B ≤ 7 sequences!
```

With K=8→1 (single KV head): **56 sequences** (8× more). This is why GQA is mandatory for long-context inference.

### Q3: The Lower Bound

Loading all params (sharded, int8): `18.4e9 / (16 × 8.1e11) = 1.42 ms`

This is the **theoretical floor** for generation step time. Below this is physically impossible.

### Q4: Sharding Rules for Generation

During generation, you **cannot use**:

- ❌ FSDP — ICI is 9× slower than HBM; moving weights over ICI adds latency
- ❌ Data parallelism — replicates params without helping bandwidth
- ❌ Sequence parallelism — no sequence to split!

**Only option**: Model parallelism. With β = W_hbm/W_ici ≈ 9, you can shard up to `F/(B×β)` ways before ICI becomes the bottleneck. For small batches, shard all 16 chips → 1.42ms step time.

### Q5: MoE = 1920 Token Batch Needed

MoE (E=16, k=2) has 212B total params but only 31.3B activated per token. The catch:

```
B_crit_MoE = B_crit_dense × E/k = 240 × 8 = 1920 tokens
```

You need 8× more concurrent requests to be compute-bound. MoE exposes the bandwidth bottleneck even more severely than dense models.

### Q6: Expert Sharding (8×16 slice)

With 128 chips (Y=8, Z=16): each chip holds `212B/128 = 1.66B params`, taking 2ms to load. Only 14.4 GB free HBM per chip for KV cache.

### Q7: 2D Weight-Stationary Sharding

The algorithm shards `Win[DX, FYZ]` and uses AllGather + AllReduce + ReduceScatter instead of traditional 1D model parallel. Total comms:

$$T_{2D} = \sqrt{\frac{8}{N}} \cdot \frac{BD}{W_{ici}}$$

vs traditional: $T_{3D} = \frac{4BD}{3 W_{ici}}$

**2D wins when N > ~80 chips** — and crucially, `T_comms ∝ 1/√N`, so **adding more chips still helps latency even when communication-bound!**

---

## 🧠 What I Learned

**The inference hierarchy** (most important constraints to check in order):

1. **Memory**: Can the model + KV cache fit? → `params + B × KV_per_seq ≤ N × HBM`
2. **Critical batch**: Are we compute or bandwidth bound? → `B vs B_crit`
3. **Sharding**: What dimensions can we split? → prefill: anything; generation: model parallel only
4. **Latency floor**: `params / (N_chips × W_hbm)` — physically impossible to go below this

**The MoE tradeoff**: Huge total params, modest activated params, but terrible inference efficiency (needs `E/k` times bigger batches). MoE is a training trick that makes inference harder.

**2D sharding is the future**: Unlike 1D model parallel where T_comms stops getting better after ~8-16 chips, 2D sharding keeps improving as ∝ 1/√N. This is why ESTI-style 2D sharding matters at scale.

---

## 📂 Code & Notes

Full notes and solutions: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax)

**Q&A Doc**: [Google Doc](https://docs.google.com/document/d/1xqNwFtALOCplvjS7mF7yf0pwJXZ9EFJYwiDskFVUdwo/edit?usp=sharing)

**Chapter**: [JAX Scaling Book - Inference](https://jax-ml.github.io/scaling-book/inference/)

---

_Inference is simple: reduce bytes, increase batch, accept that attention will always be slow._
