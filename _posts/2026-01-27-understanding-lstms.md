---
layout: post
title: "Understanding LSTMs: A Hands-on Intuition"
date: 2026-01-27 23:55:00 +0900
description: Implementing Long Short-Term Memory (LSTM) from scratch and exploring the intuition behind its gates.
tags: deep-learning machine-learning lstm
categories: achievements
giscus_comments: false
related_posts: true
---

I recently implemented a **Long Short-Term Memory (LSTM)** network from scratch using NumPy. While the formulas can look daunting, the real beauty of LSTMs lies in their intuitive design as a "controlled memory stream."

## 🧠 The "Controlled Memory" Intuition

If you've ever struggled with the technicalities of LSTMs, I highly recommend reading [Christopher Olah's classic blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). It's the gold standard for understanding how these networks truly function.

In my implementation, I focused on making the gates "feel" like what they actually do:

- **`*` (Multiplication) is the Selector:** It decides exactly how much information flow should be allowed at any given moment.
- **`+` (Addition) is the Knowledge Adder:** It creates a "highway" for the cell state, allowing gradients to flow without vanishing—solving the key problem of traditional RNNs.
- **`Sigmoid` is the Gatekeeper:** Because it outputs between 0 and 1, it’s perfect for selecting or blocking information.
- **`Tanh` is the Featurizer:** It’s great for creating new candidate features because it normalizes data while keeping it expressive.

## 🚪 What each gate "feels" like

1. **Forget Gate:** *The Eraser.* It looks at the past and asks: "Is this still worth remembering?"
2. **Input Gate:** *The Filter.* It decides which parts of the *new* input are actually useful.
3. **Candidate Memory:** *The Writer.* It prepares the new "draft" of information using `tanh`.
4. **Output Gate:** *The Presenter.* It takes the long-term memory and decides what to show for the current time step.

---

### 💻 Implementation

You can find my full implementation, along with a detailed breakdown of the intuition, in my dedicated repository:

👉 **[LSTM Implementation on GitHub](https://github.com/YashJayswal24/deepml-solns/tree/main/lstm-network)**

This project is part of my ongoing journey through the [Deep-ML](https://www.deep-ml.com/) problem sets!
