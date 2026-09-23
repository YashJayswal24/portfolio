---
layout: page
title: Speculative Decoding from Scratch
description: PyTorch implementation and benchmarking of speculative decoding (Leviathan/Chen et al.) with a distribution-preserving rejection-sampling verifier.
img:
importance: 1
category: work
related_publications:
---

[GitHub Repository](https://github.com/YashJayswal24/speculative-decoding)

**Summary:**
Implemented speculative decoding from scratch in PyTorch and benchmarked it against a production inference engine to understand where the real speedups come from.

**Highlights:**

- Implemented the draft-then-verify algorithm with modified rejection sampling, verified via KL-divergence testing that the output distribution exactly matches standard autoregressive decoding.
- Benchmarked OPT-125M/OPT-6.7B on an A30 GPU: 1.3x speedup in a from-scratch PyTorch implementation vs. 2.8x with vLLM, attributing the gap to CUDA graph kernel-launch overhead.
