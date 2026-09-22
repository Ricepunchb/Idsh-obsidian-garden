---
title: DPO
alias: Direct Preference Optimization
publish: true
date: 2026-09-22
tags:
  - RLHF
  - Fine-Tuning
---
# DPO (Direct Preference Optimization)

## 1. 개요
[[RLHF]]는 (1) 선호 데이터로 reward model을 학습하고, (2) 그 reward model로 [[PPO]] 루프를 돌리는 2단계 과정이다. DPO는 이 KL-regularized RLHF 목적함수가 사실 **닫힌 형태의 최적해**를 가진다는 관찰에서 출발해, reward model 학습도 RL 루프도 없이 선호 데이터로 정책을 **직접** 최적화한다.

## 2. 유도
[[RLHF]]의 KL-regularized objective $\mathcal J^\text{RLHF}$는 다음과 같은 닫힌 형태의 최적해를 가진다.
$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_\text{ref}(y\mid x)\exp\left(\frac1\beta R(y)\right)
$$
이를 $R(y)$에 대해 정리하면
$$
R(y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)}+\beta\log Z(x)
$$
즉, 정책 $\pi^*$와 참조 모델 $\pi_\text{ref}$의 로그 확률 비율만으로 reward를 역으로 표현할 수 있다.

## 3. DPO Loss
이 $R(y)$의 표현을 [[RLHF]]의 Bradley-Terry 손실 $P(y_w\succ y_l)=\sigma(R(x,y_w)-R(x,y_l))$에 대입하면, 두 응답에 공통으로 등장하는 $\beta\log Z(x)$가 서로 상쇄되어 사라진다. 남는 것은 관측된 선호에 대한 negative log-likelihood뿐이다.
$$
\mathcal L^\text{DPO}(\theta)=-\mathbb E_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right]
$$

>[!attention] DPO가 없앤 것들
> 별도의 reward model 학습, rollout 샘플링, [[PPO]]의 클리핑·advantage 계산이 모두 사라지고, 선호 데이터 $(x,y_w,y_l)$에 대한 **하나의 분류 손실**로 정책을 직접 학습한다. $\pi_\theta$ 자체가 암묵적으로(implicitly) reward model 역할을 겸한다고 볼 수 있다.

## 4. 한 줄 요약
> DPO는 KL-regularized RLHF 목적함수의 닫힌 해로부터 reward를 정책의 로그 확률 비율로 다시 써서, reward model 학습과 PPO 루프 없이 선호 데이터에 대한 하나의 분류 손실로 정책을 직접 최적화하는 방법이다.
