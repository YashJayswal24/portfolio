---
layout: post
title: "Scaling LLMs: From Alchemy to Science (Part 0)"
date: 2026-02-02 23:00:00 +0900
description: Understanding strong scaling, roofline models, and why Transformers won the hardware compatibility game.
tags: jax tpu systems-engineering llm scaling hardware
categories: engineering
giscus_comments: false
related_posts: true
---

I've started working through **["How to Scale Your Model: A Systems View of LLMs on TPUs"](https://jax-ml.github.io/scaling-book/)** — a book that aims to turn ML training from alchemy into science.

This post covers **Part 0: Introduction**, where I explored the fundamental trade-offs of parallelizing models across multiple chips.

---

## 🎯 The Core Question

**Why doesn't adding more chips always make training faster?**

The answer lies in understanding three bottlenecks:
1. **Compute** (Peak FLOPs)
2. **Memory** (HBM Bandwidth)
3. **Communication** (Interconnect Bandwidth)

---

## 📝 My Learning Journey: 6 Key Questions

### Round 1: Understanding the Fundamentals

#### Q1: The "Strong Scaling" Trade-off
**Question**: Why does adding more chips not always result in a linear increase in speed?

**My Answer**: Strong scaling means adding chips to reduce time for a **fixed problem size**. However, as you shard a model further, **communication becomes the bottleneck**.

**Key Insight**: 
- **Weak Scaling** = More chips → More users/data.
- **Strong Scaling** = More chips → Less time (for the same workload).
- As the **Compute-to-Communication ratio** drops, you hit a wall.

---

#### Q2: The Bottleneck Theory
**Question**: What are the three primary factors that form the "roofline"?

**My Answer**: The three pillars are:
1. **Peak FLOPs** (compute capacity)
2. **HBM Bandwidth** (memory bandwidth)
3. **Interconnect Bandwidth** (communication between chips)

**Key Insight**: The "roofline model" helps you identify which of these three is limiting your performance.

---

#### Q3: Architecture vs. Hardware
**Question**: Why is understanding interconnects more important today than five years ago?

**My Answer**: Algorithms have consolidated into **Transformers**, which are built entirely on matrix multiplications—exactly what TPUs/GPUs excel at. At high scale, any latency in the interconnect adds up.

**Key Insight**: Modern ML research is less about "algorithmic cleverness" and more about **hardware efficiency**. The **Memory Wall** is now the primary design constraint.

---

### Round 2: Deeper Systems Thinking

#### Q4: The "Alchemy" vs. "Science" Distinction
**Question**: What distinguishes ML "alchemy" from ML "science"?

**My Definition**: 
- **Science** = Predicting performance and resource usage accurately.
- **Alchemy** = Relying on intuition where things "just work."

**Key Insight**: Understanding the systems view lets us move from black-box experimentation to principled design.

---

#### Q5: The "Communication-Bound" Nightmare
**Question**: Why can doubling chips sometimes *slow down* training?

**My Answer**: The bottleneck shifts to the **ICI (Inter-Chip Interconnects)**. If communication takes longer than computation, we become **communication-bound**, and strong scaling breaks.

**Key Insight**: Adding chips increases **communication overhead**. If the interconnect bandwidth can't keep up, you're just wasting resources.

---

#### Q6: Hardware-Driven Design
**Question**: Why did Transformers "win" over other architectures?

**My Answer**: Transformers were designed for **parallelism**. Their math (attention as matrix ops) perfectly aligns with TPU/GPU capabilities, allowing them to scale with compute.

**Key Insight**: This isn't just about better language understanding—it's about **scaling efficiently** on modern hardware.

---

## 🧠 Personal Reflection

At Samsung, when we talk about "optimizing models," we're really asking:
1. Are we **compute-bound**? (Waiting for matrix ops)
2. Are we **memory-bound**? (Waiting for data to load)
3. Are we **communication-bound**? (Waiting for chips to sync)

The days of "just throw more GPUs at it" are over. If you don't account for these bottlenecks, you're just burning money.

---

## � Code & Notes

All my notes and future implementations are in the [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax) repository.

**Next up**: Part 1 - Roofline Analysis

---

*Turning alchemy into science, one section at a time.*
