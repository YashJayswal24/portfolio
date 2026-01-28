---
layout: post
title: "Pegasos Kernel SVM: The Hardest Math So Far"
date: 2026-01-28 17:45:00 +0900
description: Implementing a deterministic Pegasos algorithm for kernel SVMs and diving deep into the mathematics of support vector machines.
tags: machine-learning svm kernel-methods python
categories: achievements
giscus_comments: false
related_posts: true
---

I just tackled the **Pegasos Kernel SVM** problem on [Deep-ML](https://www.deep-ml.com/), and I have to say—this was by far the **hardest concept mathematically** I've encountered so far.

## 🧮 Two Worlds of SVM

There are two main approaches to training SVMs:

1.  **Primal (Pegasos):** Uses sub-gradient descent on the hinge loss directly.
2.  **Dual (SMO):** Uses Lagrange multipliers to solve a constrained optimization problem.

Both are math-heavy and require a solid understanding of optimization theory.

## 💡 What I Learned

-   **Lagrangian Multipliers:** The dual formulation of SVM relies on Lagrangians for constrained optimization—a concept I mostly saw in physics before!
-   **The Kernel Trick:** Allows us to compute dot products in higher-dimensional spaces without ever going there explicitly.
-   **A Note on the Deep-ML Solution:** The problem description uses `α_i ← α_i + η_t(y_i - λ*α_i)`. I found it clearer to separate the decay term: `α_i ← (1 - 1/t) * α_i + η_t * y_i`, which achieves the same mathematical result.

### 📚 Best Resource

I found this MIT resource invaluable: [An Idiot's Guide to SVMs](https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf). Don't let the title fool you—it's a comprehensive guide that requires serious effort!

---

## 💻 The Implementation

Here's my full Pegasos Kernel SVM implementation:

```python
import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    """
    n_samples, n_features = data.shape
    alphas = np.zeros(n_samples)
    bias = 0.0
    
    def linear_kernel(x1, x2):
        return np.sum(x1 * x2, axis=1)
    
    def rbf_kernel(x1, x2, sigma):
        return np.exp(-np.linalg.norm(x1 - x2, axis=1) ** 2 / (2 * sigma ** 2)).reshape(-1)
    
    kernel_fun = linear_kernel if kernel == 'linear' else lambda x1, x2: rbf_kernel(x1, x2, sigma)
    
    for t in range(1, iterations + 1):
        lr = 1 / (lambda_val * t)
        for i in range(n_samples):
            pred = np.sum(alphas * labels * kernel_fun(data[i].reshape((1, -1)), data)) + bias
            if labels[i] * pred < 1:
                alphas[i] = (1 - 1/t) * alphas[i] + lr * labels[i]
                bias += lr * labels[i]
    
    return alphas.tolist(), bias
```

### Test Results

```python
# Linear Kernel
alphas, b = pegasos_kernel_svm(
    np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), 
    np.array([1, 1, -1, -1]), 
    kernel='linear', lambda_val=0.01, iterations=100
)
print(([round(a, 4) for a in alphas], round(b, 4)))
# Output: ([100.0, 0.0, -100.0, -100.0], -937.4755)

# RBF Kernel
alphas, b = pegasos_kernel_svm(
    np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), 
    np.array([1, 1, -1, -1]), 
    kernel='rbf', lambda_val=0.01, iterations=100, sigma=0.5
)
print(([round(a, 4) for a in alphas], round(b, 4)))
# Output: ([100.0, 99.0, -100.0, -100.0], -115.0)
```

---

👉 **[Full Code on GitHub](https://github.com/YashJayswal24/deepml-solns/tree/main/pegasos-kernel-svm)**

*On to the next challenge!*
