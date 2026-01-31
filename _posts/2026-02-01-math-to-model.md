---
layout: post
title: "Math to Model: GRPO, PCA Augmentation, and SVD"
date: 2026-02-01 00:10:00 +0900
description: Implementing the GRPO objective from DeepSeekMath, AlexNet's PCA color augmentation, and SVD from scratch.
tags: deep-learning reinforcement-learning linear-algebra computer-vision python
categories: achievements
giscus_comments: false
related_posts: true
---

I've been diving deep into some challenging problems on [Deep-ML](https://www.deep-ml.com/), ranging from classical linear algebra to modern reinforcement learning. Here are my solutions and takeaways for three distinct problems.

## 1. GRPO Objective Function (Reinforcement Learning)

Implementing the **Group Relative Policy Optimization (GRPO)** objective from the DeepSeekMath paper was a great refresher on RL math.

**Key Insight:**
The equations look intimidating, but it's really just a loss function designed to control weight updates. It combines a PPO-style clipped surrogate with a KL divergence penalty to keep the model from "going berserk."

👉 **[Code Solution](https://github.com/YashJayswal24/deepml-solns/tree/main/grpo-objective)**

## 2. PCA Color Augmentation (Computer Vision)

This is the technique used in the legendary **AlexNet**. Instead of random noise, we use **PCA** to find the principal colors in an image and add noise along those semantic axes.

**Key Insight:**
It's an elegant way to simulate lighting intensity and color temperature changes while preserving the image's semantic content.

👉 **[Code Solution](https://github.com/YashJayswal24/deepml-solns/tree/main/pca-color-augmentation)**

## 3. SVD of a 2x2 Matrix (Linear Algebra)

**Singular Value Decomposition (SVD)** is the grandfather of dimensionality reduction.

**Key Insight:**
Implementing this from scratch brought back memories of college linear algebra. It's beautiful how the eigenvectors of $A^T A$ give us the right singular vectors of $A$.

👉 **[Code Solution](https://github.com/YashJayswal24/deepml-solns/tree/main/svd-2x2)**

---

*Solving these problems is the best way to keep the mathematical intuition sharp!*
