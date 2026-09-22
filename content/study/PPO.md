---
title: PPO
alias: Proximal Policy Optimization
publish: true
date: 2026-09-22
tags:
  - RLHF
  - RL
---
# PPO (Proximal Policy Optimization)

## 1. 개요
[[Policy Gradient]]의 off-policy surrogate objective에 importance ratio **clipping** 메커니즘을 더해 학습 안정성을 확보한 알고리즘이다. [[GRPO]]는 PPO의 클리핑 메커니즘을 그대로 물려받으면서 critic(value function)만 그룹 상대 advantage로 대체한 것이다.

>[!abstract] 핵심 아이디어
> "같은 배치의 rollout으로 여러 번 gradient step을 밟고 싶다 (효율성) — 하지만 정책이 rollout을 만든 $\pi_{\theta_\text{old}}$에서 너무 멀어지면 surrogate objective의 근사가 깨진다. 그러니 정책이 이미 충분히 좋은 방향으로 움직였다면 그 이상은 보상을 주지 않도록 잘라내자(clip)."

## 2. Clipped Surrogate Objective
[[Policy Gradient]]에서 정의한 importance ratio $r_t=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$를 그대로 사용한다. PPO는 baseline으로 value function $V_\psi$를 사용하므로, $R(\tau)$ 대신 시점별 advantage $A_t=R(\tau)-V_\psi(s_t)$를 쓴다.

기본 surrogate objective:
$$
\mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_tr_tA_t\right]
$$

clipping을 적용한 목적함수는 $r_t$를 $[1-\epsilon,1+\epsilon]$ 범위로 잘라낸 값과 원래 값 중 **더 작은 쪽**을 취한다.
$$
\mathcal J^\text{CLIP}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\min(r_tA_t,\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right]
$$
- 클리핑은 같은 rollout 배치로 여러 gradient step을 밟아도 정책이 불안정해지지 않도록 해준다.
- unbiasedness를 일부 포기하는 대신 안정적인 업데이트를 얻는 것이다.
- 클리핑된 목적함수는 $\pi_{\theta_\text{old}}$에서 많이 벗어나지 않은 한에서만 좋은 근사가 된다.

## 3. 클리핑이 활성화되는 4가지 경우
$\min(\cdot,\cdot)$ 때문에 실제로 클리핑이 gradient에 영향을 주는 경우는 아래처럼 나뉜다.

| $r_t$ 조건 | $A_t$ | 실제 사용되는 값 | $\theta$에 대한 gradient | 해석 |
| :--- | :--- | :--- | :--- | :--- |
| $r_t>1+\epsilon$ | $A_t>0$ | $(1+\epsilon)A_t$ | 0 | 이미 좋은 행동을 충분히 밀었으니 더 밀지 않는다 |
| $r_t<1-\epsilon$ | $A_t<0$ | $(1-\epsilon)A_t$ | 0 | 이미 나쁜 행동을 충분히 눌렀으니 더 누르지 않는다 |
| $r_t>1+\epsilon$ | $A_t<0$ | $r_tA_t$ (clip 안 됨) | $\ne 0$ | 확률을 올렸는데 알고 보니 나쁜 행동 — 계속 눌러서 교정 |
| $r_t<1-\epsilon$ | $A_t>0$ | $r_tA_t$ (clip 안 됨) | $\ne 0$ | 확률을 낮췄는데 알고 보니 좋은 행동 — 계속 올려서 교정 |

>[!attention] 클리핑은 비대칭적이다
> 클리핑은 "정책이 이미 advantage가 가리키는 방향으로 움직인 경우"에만 발동해서 그 이상 가는 것을 막는다. 반대로 정책이 실수로 잘못된 방향(advantage와 반대)으로 움직인 경우에는 $r_t$가 $1+\epsilon$을 넘거나 $1-\epsilon$ 아래로 내려가도 클리핑되지 않고 gradient가 그대로 흘러 교정을 허용한다.

## 4. 학습 루프
PPO는 (1) 현재 정책으로 rollout 배치를 수집한 뒤, (2) 같은 배치에 대해 clipped surrogate objective로 **여러 번** gradient step을 밟는다. 이것이 PPO가 off-policy 알고리즘인 이유 — 두 번째 step부터는 이미 $\pi_\theta\ne\pi_{\theta_\text{old}}$이기 때문이다. 클리핑이 바로 이 반복 업데이트에서 정책이 $\pi_{\theta_\text{old}}$로부터 너무 멀어지지 않도록 잡아주는 역할을 한다.

## 5. 한 줄 요약
> PPO는 [[Policy Gradient]]의 off-policy surrogate objective에서, 정책이 advantage 방향으로 이미 충분히 움직였다면 importance ratio $r_t$를 $[1-\epsilon,1+\epsilon]$로 잘라 더 이상 보상을 주지 않는 방식으로, 같은 rollout 배치를 여러 번 재사용하면서도 학습을 안정시킨다.
