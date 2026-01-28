---
layout: post
title: "DenseNet Block: Brute-Force Feature Reuse"
date: 2026-01-28 18:50:00 +0900
description: Implementing a DenseNet dense block from scratch and comparing it to ResNet.
tags: deep-learning cnn densenet resnet python
categories: achievements
giscus_comments: false
related_posts: true
---

I implemented a **DenseNet Dense Block** from scratch on [Deep-ML](https://www.deep-ml.com/). DenseNet's core idea is simple: concatenate *all* previous feature maps into each layer.

## 💡 Quick Learnings

- **Type Matters:** Using `0` instead of `0.0` in `np.maximum` caused type conversion bugs. Details matter!
- **DenseNet Philosophy:** Throw all the compute you have at the problem. Concatenate everything.

## ⚖️ DenseNet vs. ResNet

| Feature | DenseNet | ResNet |
|---|---|---|
| Connection | Concatenation | Addition |
| Parameters | Fewer | More |
| GPU Memory | **High** | Low |
| Training Speed | **Slow** | Fast |
| Best For | Max accuracy | Practical deployment |

**Verdict:** Same goal (gradient flow), but ResNet is more practical.

---

## 💻 The Implementation

```python
import numpy as np

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    N, H, W, C0 = input_data.shape
    
    def relu(x):
        return np.maximum(0.0, x)
    
    def conv2d(X, kernel, stride=1, padding='zero'):
        n_b, n_h, n_w, n_c = X.shape
        k_h, k_w, c_in, c_out = kernel.shape
        if padding == 'zero':
            pad_h = (k_h - 1) // 2
            pad_w = (k_w - 1) // 2
            X_padded = np.pad(X, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        out = np.zeros((n_b, n_h, n_w, c_out))
        for b in range(n_b):
            for h in range(n_h):
                for w in range(n_w):
                    for c in range(c_out):
                        h_strt = h * stride
                        h_end = h_strt + k_h
                        w_strt = w * stride
                        w_end = w_strt + k_w
                        out[b, h, w, c] = np.sum(X_padded[b, h_strt:h_end, w_strt:w_end, :] * kernel[:, :, :, c])
        return out
    
    out = input_data
    for lay_idx in range(num_layers):
        out = relu(input_data)
        out = conv2d(out, kernels[lay_idx])
        input_data = np.concatenate((input_data, out), axis=-1)
    return input_data
```

👉 **[Full Code on GitHub](https://github.com/YashJayswal24/deepml-solns/tree/main/densenet-block)**
