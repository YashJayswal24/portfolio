---
layout: page
title: Panic-Driven Crowd Dynamics
description: Developed strategies to reduce death count in dense crowd panic situations through simulation.
img: assets/img/12.jpg
importance: 2
category: work
related_publications: 
---

[GitHub Repository](https://github.com/YashJayswal24/crowd_evacuation) | [Project Report](https://drive.google.com/file/d/1AKV16C9QtdNNqSU6K-GetxuzQnaQ_GYw/view?usp=sharing)

# A simple model for panic driven motion in a refugee camp

**Authors:** Andrea Neenu Jose, Yash Jayswal, and Sujin B Babu  
**Institution:** Out of Equilibrium Group, Department of Physics, Indian Institute of Technology Delhi  
**Date:** October 17, 2025

---

### Abstract
The increasing number of refugees due to natural and man-made disasters presents significant challenges, particularly in ensuring the safe and efficient distribution of food in refugee camps. Panic-driven motion during food distribution often leads to overcrowding, collisions, and fatalities. In this study, we model the food distribution process using a crowd dynamics framework to analyze how movement patterns contribute to injuries and deaths. Our simulations show that as food distribution begins, refugees rush toward the food distribution point, leading to high-density congestion and dangerous interactions. We explore strategies to mitigate these risks by introducing fixed obstacles near the food distribution point. The results demonstrate that a single obstacle can reduce fatalities by approximately 15%, while an optimized two-obstacle configuration achieves up to a 70% reduction. Strategic obstacle placement offers a practical and low-cost intervention for improving safety in refugee camps.

---

### Introduction
Natural calamities are increasing due to the effects of climate change, often forcing millions to shift to refugee camps. A major challenge is providing food and water to starving individuals. Once food arrives, panic often sets in, leading to competitive and pushing behavior. These situations cause impatient refugees to suffer fatal injuries. 

In this work, we use an improved version of the Social Force Model (SFM) proposed by Helbing to simulate crowd behavior in panic situations. The SFM accounts for both physical and psychological interactions, enabling it to reproduce self-organization phenomena like arching and lane formation. Our study focuses on replicating the "faster is slower" effect, where increased competitiveness leads to excessive crowd pressure and obstruction.

### Model
We assume $N$ refugees to be spheres with a mass $m_i$ randomly distributed in a 2D box. These refugees initially move towards a food distribution point $P_0$. The equation of motion for each refugee is given by:

$$m_i \frac{d\vec{v}_i(t)}{dt} = m_i \frac{v_i^0 \vec{e}_i^0(t) - \vec{v}_i(t)}{\tau_i} + \sum_{i \neq j} \vec{f}_{ij} + \sum_{W} \vec{f}_{iW}$$

Refugees are randomly spawned around the food distribution point. Once they collect food, they exit in designated directions to minimize collisions with the incoming crowd. We set a maximum pressure threshold beyond which a refugee is considered immobile (injured).

---

### Results and Discussion

#### Simulation Videos

Our simulations demonstrate the effectiveness of strategic obstacle placement in reducing fatalities during panic-driven crowd dynamics.

<div class="row mt-3">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.liquid path="https://youtube.com/shorts/Xe6k2Pxfrcs" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>No Obstacle Case:</strong> High congestion and fatalities as refugees rush toward the food distribution point without guidance.
        </div>
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.liquid path="https://youtube.com/shorts/X56qHXFP3AA" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>One Obstacle Case:</strong> Single obstacle reduces fatalities by ~15% by redirecting crowd flow.
        </div>
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm-8 mx-auto mt-3 mt-md-0">
        {% include video.liquid path="https://youtube.com/shorts/buY6AbsJ4Yo" class="img-fluid rounded z-depth-1" %}
        <div class="caption">
            <strong>Two Obstacle Case:</strong> Optimized two-obstacle configuration achieves up to 70% reduction in deaths by splitting crowd into three streams.
        </div>
    </div>
</div>

#### No Obstacle Case
In this scenario, refugees rush towards the food distribution point without any guidance. As crowd density grows, we observe a sharp increase in deaths due to collisions and high pressure. Our analysis revealed that the number of deaths reduces as the Food Distribution Radius (FDR) increases, providing more space for collection. However, very large FDRs are often not practically viable. A significant cause of fatalities is the collision between incoming refugees and those retreating after receiving food.

#### One Obstacle Case
To reduce injuries, we introduced a single circular obstacle near the food distribution point. This redirects the crowd, eases pressure, and guides people more smoothly toward the distribution point. We found that placing the obstacle slightly asymmetrically and at an optimal distance can reduce deaths by approximately 15%. If the obstacle is too small or too far, its effectiveness diminishes.

#### Two Obstacle Case
Further testing with two obstacles showed even more significant improvements. This configuration splits the incoming crowd into three distinct streams, reducing turbulence and pressure in the central region. With two obstacles, we achieved up to a 70% reduction in fatalities compared to the no-obstacle case. The gap between obstacles and their size are critical parameters for optimal flow.

---

### Conclusion
Our findings highlight the critical role of strategic obstacle placement in managing panic-driven crowd dynamics. While a single obstacle provides a 15% reduction in fatalities, an optimized two-obstacle system can save up to 70% more lives. This practical, low-cost intervention can be implemented in refugee camps to enhance safety during food and water distribution. Future work could explore multiple distribution points and varying obstacle shapes.
