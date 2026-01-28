---
layout: post
title: "Building a Primitive GPT-2: Layers & Dimensions"
date: 2026-01-28 15:45:00 +0900
description: Merging deep learning concepts to build a simplified GPT-2 text generator and understanding the importance of dimensions.
tags: deep-learning nlp transformers gpt-2 python
categories: achievements
giscus_comments: false
related_posts: true
---

I recently took on the challenge of implementing a simplified **GPT-2 Text Generation** function on [Deep-ML](https://www.deep-ml.com/). This exercise was a fantastic way to bring together separate concepts—embeddings, layer normalization, and autoregression—into one cohesive system.

## 🧩 The Puzzle Pieces

Building a generator like this forces you to confront the reality of the architecture:

1.  **Embeddings:** Combining word identity with position identity (`wte + wpe`).
2.  **Layer Norm:** Stabilizing the features before projection.
3.  **The Loop:** Autoregressively predicting the next token and feeding it back.

## 💡 Lightbulb Moments

### Dimensions are Everything
I used to get confused by the endless reshaping in PyTorch/NumPy. But during this project, I realized that **noting down the dimensions** (like `(seq_len, d_model)`) makes everything fall into place. It turns abstract math into a shape-matching puzzle.

### Layer Norm vs. Batch Norm
I got stuck on which axis to normalize for a while. Then it clicked: **Layer Norm gives every feature a fair shot.** Unlike Batch Norm which looks across the batch, Layer Norm standardizes the features for a *single example*, ensuring that no single feature dominates the gradients.

### The Cost of "Memory-less" Generation
Writing the generation loop manually really highlights the inefficiency. To generate the 5th token, we re-process the first 4. This redundancy is exactly why **KV Caching** is a critical optimization in modern LLM serving!

---

### 💻 The Code

You can see my simplified implementation here:

👉 **[GPT-2 Implementation on GitHub](https://github.com/YashJayswal24/deepml-solns/tree/main/gpt2-text-generation)**

It uses a dummy encoder and random weights, but the logic is identical to the real thing!
