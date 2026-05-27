---
layout: post
title: "IMC Prosperity 4: Top 1.2% Out of 22,000+ Teams"
date: 2026-05-27 12:00:00 +0900
description: How our team Gradient Exploders placed #266 overall and #61 in manual trading at the IMC Prosperity 4 algorithmic trading challenge.
tags: algorithmic-trading quantitative-finance game-theory python competition
categories: achievements
giscus_comments: false
related_posts: false
---

A few months ago, my teammates [Gangula Bhuvan Reddy](https://www.linkedin.com/in/gangula-bhuvan-reddy/) and [Andrea Jose](https://www.linkedin.com/in/andreajose/) and I entered the [IMC Prosperity 4](https://prosperity.imc.com/) algorithmic trading challenge. We went in with modest expectations — more curious about the problem space than chasing a podium finish. We ended up performing far better than we anticipated.

## 🏆 Results

Out of **22,000+ participating teams** worldwide, our team **Gradient Exploders** placed:

- 🥇 **#266 Overall** — Top **1.2%** globally
- 🎯 **#61 in Manual Trading** — Top globally
- JP **#2 in Country**
- 💰 **Team Total: 485,957** SeaShells (target was 200,000 — we more than doubled it)

{% include figure.liquid loading="eager" path="assets/img/imc_round_2_result_pic.jpeg" class="img-fluid rounded z-depth-1" caption="Gradient Exploders — Round 2 Leaderboard" %}

We're also in the **top 15% of teams that qualified for the finals**, so the competition isn't over yet.

---

## 🎮 What Is IMC Prosperity?

IMC Prosperity is a multi-round algorithmic trading simulation run by [IMC Trading](https://www.imc.com/). Each round introduces new financial instruments (products), and teams compete by writing a Python `Trader` class that submits orders against a live order book. There are two components per round:

- **Algorithmic Trading** — Your bot trades against a simulated market. You submit Python code that runs at every timestamp, making buy/sell decisions on the order book.
- **Manual Trading** — Game-theoretic puzzles where you analyze market scenarios and make one-shot decisions to maximize profit.

The challenge spans several rounds with increasing complexity: from simple mean-reversion on single products to multi-asset arbitrage, conversion arbitrage, and coordination games.

---

## 🧠 Our Approach

### Algorithmic Trading

Our most successful algorithmic strategy was a **Kalman filter-based adaptive market maker** for `INTARIAN_PEPPER_ROOT`. Rather than using a fixed fair-value estimate, we continuously updated our price beliefs using the Kalman filter to adapt to market drift:

```python
# Simplified core of our Kalman adaptive strategy
def kalman_update(self, mid_price: float) -> float:
    # Predict step
    self.x_est = self.x_est  # no drift model
    self.p_est = self.p_est + self.process_noise

    # Update step
    k_gain = self.p_est / (self.p_est + self.obs_noise)
    self.x_est = self.x_est + k_gain * (mid_price - self.x_est)
    self.p_est = (1 - k_gain) * self.p_est

    return self.x_est
```

The key insight was that **simple, structurally sound ideas consistently outperformed complex models**. When we tried to layer on more sophisticated signals (momentum, regime detection), performance degraded — not because the ideas were wrong, but because the backtesting data wasn't rich enough to tune them reliably.

### Manual Trading

The manual rounds were pure **game theory**. Each puzzle presented a scenario (e.g., a bidding war, a prisoner's dilemma-style coordination game, or an options payoff matrix) and asked you to commit to a strategy. Our **#61 global manual rank** was the highlight of the competition — it validated that thinking probabilistically and modeling other participants' reasoning was more valuable than trying to find the "mathematically optimal" answer in isolation.

---

## 🛠️ The Backtester

To iterate on strategies quickly, we built our own **Python backtester** on top of [jmerle's Prosperity 3 backtester](https://github.com/jmerle/imc-prosperity-3-backtester), rewriting it in a more modular OOP style for Prosperity 4's data format.

**GitHub:** [YashJayswal24/imc-prosperity-4-backtester](https://github.com/YashJayswal24/imc-prosperity-4-backtester)

The backtester has three core components:

| Component | Role |
|---|---|
| `BackTester` | Top-level driver — loads algorithm, iterates over rounds/days, merges results |
| `TestRunner` | Daily simulator — replays each timestamp, feeds `TradingState` to your algorithm |
| `OrderMatchMaker` | Simulates exchange mechanics — fills orders against the historical order book |

**Usage:**
```bash
# Run your algorithm against round data
python -m prosperity4bt algorithms/Round_2/trader_r1_kalman_adaptive_final.py 2

# Run against a specific day
python -m prosperity4bt algorithms/Round_2/trader_r1_kalman_adaptive_final.py 2--1
```

Having a local backtester was critical — the official platform only gives you feedback after submitting, so the ability to run hundreds of iterations locally and inspect per-timestamp P&L was invaluable.

---

## 💡 Key Takeaways

1. **Simplicity wins.** Every time we added complexity (multi-signal models, adaptive parameters, regime detection), we were more likely to overfit than to improve. The strategies that worked best had a clean, interpretable hypothesis.

2. **Backtesting is a tool, not a truth.** Backtest results are necessary but not sufficient. We learned to treat strong backtest performance with skepticism and focus on whether the *reasoning* behind a strategy held up.

3. **Game theory is underrated in algo trading.** The manual rounds made it clear: understanding *how other participants reason* is as important as finding the correct expected value. This applies to algorithmic trading too — if everyone runs the same strategy, the edge disappears.

4. **Community resources are invaluable.** Reading through open-source solutions from Prosperity 2 and 3 gave us a mental model of the problem space before Round 1 even started.

---

## 🔭 What's Next

We've qualified for the finals and are looking forward to seeing how the competition evolves. The later rounds tend to introduce more complex instruments and deeper market microstructure — exactly the kind of problem space where we want to improve.

If you're considering entering IMC Prosperity, I'd highly recommend it. The problems are well-designed, the community is active, and it's one of the best practical environments to apply probability theory and market microstructure concepts outside of a live trading environment.
