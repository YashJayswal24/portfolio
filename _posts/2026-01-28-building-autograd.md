---
layout: post
title: "Building Autograd: Chain Rule and Topo-Sort"
date: 2026-01-28 00:05:00 +0900
description: Implementing a basic autograd engine from scratch and discovering the real-world power of topological sorting.
tags: deep-learning autograd micrograd python
categories: achievements
giscus_comments: false
related_posts: true
---

I recently tackled the "Basic Autograd Operations" problem on [Deep-ML](https://www.deep-ml.com/), and it was an eye-opening experience. Following the footsteps of Andrej Karpathy (check out his [legendary Micrograd video](https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg)), I built a scalar-valued autograd engine.

## 🌉 Bridging the Gap: Theory to Code

What I loved most about this project was how systematic the commands felt. The chain rule isn't just a math formula here; it's a series of small, local instructions that each `Value` object knows how to execute.

### 🔄 Topological Sorting in the Wild

For a long time, I only saw **Topological Sorting** in competitive programming contexts (LeetCode/CodeForces). But in an autograd engine, it's the "secret sauce." It ensures that before we compute the gradient for any node, we've already finished with everything that depends on it. 

This simple algorithm turns a messy graph of operations into a perfectly ordered sequence for backpropagation.

### 🐍 The Elegance of Python

The simplicity with which we can write this in Python is stunning. One single `backward()` function can set all gradients across the entire network, mirroring the core functionality of heavy-weights like PyTorch.

---

### 💻 The Implementation

Here is my `Value` class that handles the heavy lifting:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0 
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1  * out.grad
            other.grad += 1  * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
```

Check out the full repository and tests here: 
👉 **[Autograd Project on GitHub](https://github.com/YashJayswal24/deepml-solns/tree/main/autograd-operations)**

---
*Next up: Building actual neurons using this autograd engine!*
