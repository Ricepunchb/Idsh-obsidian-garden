---
title: Markov Decision Process
alias: MDP
publish: true
tags:
  - AI
  - RL
date: 2024-07-05
---


- 다음 State는 현재 State와 Action에 의해서만 결정되며 그 이전의 State, Action과는 무관
- "**Transition Function**" 전이함수 : $$s_{t+1} \sim P\{s_{t+1}|s_t,a_t\} = P\{s_{t+1}|(s_t,a_t),(s_{t-1},a_{t-1}),...,(s_0,a_0)\}$$
- "**Reward Function**" 보상함수 : $$r_t = R(s_t,a_t,s_{t+1})$$
- MDP의 구성은 상태집합, 행동집합, 전이함수, 보상함수

- "**Return**" 이득 : 어떤 경로 $\tau = (s_0,a_0,r_0),(s_1,a_1,r_1),...,(s_n,a_n,r_n)$을 통해 학습이 진행되었을 때, Agent가 받는 할인율이 적용된 보상의 합 $$R(\tau) = r_0+\gamma r_1 + \gamma^2 r_2 + \cdots + \gamma^nr_n = \sum^n_{i=0}\gamma^ir_i$$
- "**Discount**" 할인률 $\gamma$ : $0\le\gamma\le1$ ,  미래 보상에 대한 현 시점에서의 할인. 미래보단 현재 보상을 더 중요시함
- "**Objective Function**" 목적함수 : 어떤 정책 $\pi$를 따라 행동이 결정됬을 때 얻어지는 Return의 기댓값 $$J(\pi) = E_{\tau\sim\pi}[R(\tau)]$$
- 전이함수와 보상함수를 모르는 상태에서 경험적으로 얻은 $(s_t,a_t,s_{t+1})$를 통해 함수를 추정하고 이득을 최대로 하는 최선의 정책을 찾는 것
