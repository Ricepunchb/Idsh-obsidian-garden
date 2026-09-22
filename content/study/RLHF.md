---
title: RLHF
alias: Reinforcement Learning from Human Feedback
publish: true
date: 2026-09-22
tags:
  - RLHF
  - Fine-Tuning
---
# RLHF (Reinforcement Learning from Human Feedback)

## 1. 개요
[[Instruction Tuning]] 이후, 사람의 선호(preference)에 맞춰 모델을 정렬(alignment)하는 단계에서 쓰이는 방법이다. [[PPO]]를 사용해 정책 $\pi_\theta$를 최적화하되, 참조 모델 $\pi_\text{ref}$(보통 SFT 직후의 모델)에서 너무 멀어지지 않도록 KL penalty를 추가한다.
$$
\mathcal J^\text{RLHF}(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[R(\tau)-\beta D_{\text{KL}}(\pi_\theta\mid\mid\pi_\text{ref})\right]
$$
- 실무에서는 이 KL penalty를 시퀀스 전체가 아니라 **토큰 단위**로 계산해서 매 토큰의 reward에 합쳐 넣는다 (per-token reward에 fold).
- $\beta$가 클수록 정책이 $\pi_\text{ref}$ 근처에 머무르도록 강하게 규제한다 — reward를 높이려다 참조 모델과 동떨어진, 부자연스러운 텍스트를 생성하는 reward hacking을 방지하는 역할이다.

## 2. Reward Model 학습 (Bradley-Terry)
사람의 선호 데이터는 보통 "두 응답 중 어느 쪽이 더 나은가"라는 순위(pairwise comparison) 형태로 수집된다. 이를 스칼라 reward로 바꾸기 위해 **Bradley-Terry 모델**을 가정한다: 응답 $y_w$가 $y_l$보다 선호될 확률은
$$
P(y_w\succ y_l)=\frac{\exp(R(x,y_w))}{\exp(R(x,y_w))+\exp(R(x,y_l))}=\sigma(R(x,y_w)-R(x,y_l))
$$
LM 위에 분류 head를 얹어 다음을 최소화하도록 reward model $\varphi$를 학습한다.
$$
\mathcal L(\varphi)=-\log P(y_w\succ y_l)
$$
- 이는 참 라벨이 항상 $y=1$이고 $p=P(y_w\succ y_l)$인 이진 교차 엔트로피 손실 $-(y\log p+(1-y)\log(1-p))$과 동일한 형태다.

>[!tip] GRPO / rule-based reward와의 관계
> [[GRPO]]는 별도의 reward model 없이 규칙 기반(rule-based) reward를 쓰는 경우가 많지만, RLHF의 원형은 위처럼 **선호 데이터로 학습한 reward model + PPO** 조합이다. reward model 학습 자체를 아예 생략하고 선호 데이터로 정책을 직접 학습하는 방법이 [[DPO]]다.

## 3. 한 줄 요약
> RLHF는 사람의 pairwise 선호 데이터를 Bradley-Terry 모델로 reward model에 녹여낸 뒤, 그 reward로 [[PPO]]를 돌리되 참조 모델과의 KL penalty로 과도한 이탈을 막는 정렬 방법이다.
