---
layout: page
title: Large Language Model Scaling
description: Systems Engineering, Distributed Computation, and Mixture of Experts on TPUs
img: assets/img/all-to-all.gif
importance: 1
category: work
related_publications: false
---

A comprehensive deep dive into the systems engineering behind Large Language Models, profiling and optimizing JAX workloads on Google TPUs to handle extreme model scaling. 

As we approach the physical limits of hardware, the bottlenecks in deep learning have shifted from pure compute (FLOPs) to memory bandwidth and interconnect networking. This project tackles those bottlenecks through direct hardware-level optimization, stepping past auto-compilers to explicitly manage multi-device communication topologies.

---

## 1. The Mathematics of Hardware Sharding

Standard data parallelism collapses when models grow beyond the High Bandwidth Memory (HBM) capacity of a single chip. I implemented and evaluated mathematically derived parallelization techniques in JAX to circumvent this memory wall.

### Fully Sharded Data Parallelism (FSDP)
In standard training, Adam optimizer states, gradients, and model parameters dictate a memory requirement of roughly $16\times$ the parameter count. FSDP solves this by sharding weights across $N$ devices. Each device holds only the slice $W_i \in \mathbb{R}^{\frac{d}{N}}$. 
During a forward pass, devices invoke a collective `AllGather` operation to reconstruct the full layer, compute the activation, and immediately discard the non-local weights. During backpropagation, gradients are synchronized and sharded via `ReduceScatter` in $O(\log_2 N)$ steps.

### Tensor Parallelism (TP)
Rather than sharding across layers, Tensor Parallelism splits individual matrix multiplications along the hardware topologies. For a linear layer $Y = XW$, if we split $W$ along its columns into $k$ shards $(W_1, W_2, \dots, W_k)$, we compute:
$$ Y_i = XW_i $$
This results in independent shard outputs. An `AllGather` is then used to concatenate the final output. Alternatively, splitting rows requires a subsequent `AllReduce` operation across devices.

---

## 2. Solving the Mixture of Experts (MoE) Memory Wall

A major challenge in scaling models like Mixtral or Grok is handling the communication overhead of the Mixture of Experts (MoE) module. 

A standard MoE layer uses a router to compute a probability distribution over $E$ experts for input $X$:
$$ P(X) = \text{Softmax}(XW_r) $$
When relying on `jax.jit` auto-sharding, the XLA compiler often defaults to massive network broadcasts. To evaluate the Top-2 experts for a batch sequence of size $S$ and model dimensionality $D$ across $N$ chips, `AllGather` forces a copy of every token to every device, resulting in an uncontrolled $O(N \times S \times D)$ memory spike per forward pass, causing immediate OOM clusters crashes.

### Advanced Optimization (`jax.shard_map` \+ All-to-All)
By dropping down to a device-level computational view using `jax.shard_map`, I eliminated the compiler's naive fallback. 

I implemented a targeted **All-to-All routing pipeline**. Each processing unit chunks its batches and selectively dispatches encoded tokens across the Inter-Chip Interconnect (ICI) *only* to the specific device housing the required expert. Memory utilization dropped back down to exactly $O(\frac{S \times D}{N})$, and I was able to successfully hide the dispatch latency entirely behind the expert FLOP calculations.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/all-to-all.gif" title="All-to-All Routing" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Visualization of efficient All-to-All token routing compared to naive token broadcasting over a mesh network.
</div>

---

## 3. XLA Tracing & Hardware Profiling

To achieve these optimizations, I relied heavily on TPU profiling tools. By injecting `jax.profiler.trace` into the code and visualizing the execution graphs in Perfetto, I diagnosed invisible compiler bottlenecks.

I utilized flags such as `XLA_FLAGS="--xla_dump_to=/tmp/"` to inspect the generated HLO (High-Level Optimizer) graphs. This allowed me to observe the exact moment when the compiler inserted unnecessary blocking synchronization primitives. 

By rewriting my custom algorithms to overlap compute and communication (e.g., executing `jax.lax.psum_scatter` as a ring reduction), I reduced the layer latency on a Kaggle TPU v5e-8 cluster from an initial **73ms** down to exactly **1.7ms**—an enormous **43x speedup**.

---

## 4. Serving at Scale 

Beyond training optimizations, I studied the deployment architecture required to serve LLMs efficiently at inference time (such as LLaMA 3). I explored KV-caching optimizations, continuous batching, and how inference servers (like vLLM and TensorRT-LLM) manage PageAttention to ensure sustained high throughput across multi-node GPU/TPU deployments.

---

## Code & Open Source Deep Dives

You can read the underlying mathematics, explore the XLA traces, and see the exact implementations of my work here:
- **GitHub Repository**: [Model_scaling_jax](https://github.com/YashJayswal24/Model_scaling_jax) 
- **Kaggle Notebook**: [MoE Scaling on TPU](https://www.kaggle.com/code/reptor420/moe-scaling-on-tpu/notebook)

### Read the Blog Series
I documented my derivations and hardware profiling step-by-step in my "How to Scale" blog series:
- [10. JAX Parallelism & MoE on TPUs](/blog/2026/jax-parallelism-moe/)
- [12. GPUs vs TPUs for LLMs](/blog/2026/gpus-for-llms/)
- [9. TPU Profiling & Roofline Analysis](/blog/2026/tpu-profiling/)
