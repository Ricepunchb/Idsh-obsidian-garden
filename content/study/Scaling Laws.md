---
title: Scaling Laws
alias: 스케일링 법칙
publish: true
date: 2026-09-22
tags:
  - LLM
  - Distributed-Training
---
# Scaling Laws

## 1. 개요
모델 크기, 데이터 양, 연산량(compute) $C$가 커질수록 손실(loss)과 최적 하이퍼파라미터가 어떻게 변하는지를 예측하는 경험적 법칙들이다. 작은 스케일에서 실험해 얻은 관계식을 바탕으로, 훨씬 큰 모델을 학습하기 전에 학습률이나 최종 손실을 미리 추정할 수 있다.

>[!abstract] 핵심 철학
> "작은 모델 여러 개를 학습시켜 얻은 곡선을 외삽(extrapolate)하면, 아직 학습하지 않은 큰 모델의 최적 하이퍼파라미터와 예상 손실을 (비싼 실험 없이) 미리 알 수 있다."

## 2. Maximal Update Parameterization ($\mu P$)
- **핵심 문제**: 작은 스케일에서 찾은 하이퍼파라미터(특히 learning rate)가 큰 스케일에서는 그대로 일반화되지 않는다.
- **원인**: 표준적인 파라미터화(standard parameterization)에서는 모델 width가 커질수록 레이어별 업데이트의 크기가 서로 다른 배율로 벌어진다. 즉 width를 키우면 레이어마다 "실질적인 학습률"이 제각각 달라지는 셈이다.
- **해결**: $\mu P$는 레이어별로 초기화와 learning rate를 조정해서, width가 바뀌어도 **weight 대비 업데이트 크기의 비율이 일정하게 유지**되도록 만든다.
- 주로 width 방향의 스케일링을 다루는 기법이다 (depth 방향은 별도 논의).

>[!tip] 실무적 의미
> $\mu P$를 적용하면 작은 width의 프록시(proxy) 모델에서 찾은 learning rate를 그대로 큰 모델에 전이(transfer)해서 쓸 수 있다 — 큰 모델로 직접 하이퍼파라미터 탐색을 할 필요가 없어진다.

## 3. Compute에 대한 피팅
### 3.1. Learning Rate ~ Compute
최적 learning rate가 compute $C$에 대해 거듭제곱(power law) 형태로 감소한다고 가정하고 피팅한다.
$$
\text{LR}(C)=\beta C^{-\alpha}\implies\log\text{LR}(C)=\log\beta-\alpha\log C
$$
- 양변에 로그를 취하면 $\log C$에 대한 선형 관계가 되어, 로그-로그 스케일에서 직선 피팅으로 $\alpha,\beta$를 구할 수 있다.

### 3.2. Loss ~ Compute
손실도 마찬가지로 compute에 대한 거듭제곱 법칙을 따르되, **irreducible loss** 항 $\mathcal L_\infty$를 반드시 포함해야 한다.
$$
\mathcal L(C)=\mathcal L_\infty+\beta C^{-\alpha}\implies \log(\mathcal L(C)-\mathcal L_\infty)=\log\beta-\alpha\log C
$$
- $\mathcal L_\infty$는 데이터 자체의 엔트로피(불확실성)에 해당한다. 이 항이 없으면 $C\to\infty$일 때 $\mathcal L\to 0$이 되어버리는데, 실제로는 데이터에 내재한 불확실성 때문에 손실이 0으로 수렴할 수 없다.

## 4. 피팅 방법: 최소제곱법 (Least Squares)
위 식들의 $\alpha,\beta$(그리고 $\mathcal L_\infty$)는 잔차 제곱합을 최소화하는 최소제곱법(least squares)으로 구한다.
$$
S=\sum_i(y_i-f(x_i))^2
$$

### 4.1. 선형 최소제곱 (Linear Least Squares)
- $\log\text{LR}$ 대 $\log C$처럼 로그를 취해 선형 관계로 바꾼 경우, 닫힌 형태(closed-form) 해가 존재한다.
- $y=X\beta$ 형태일 때 해는
$$
\beta=(X^\top X)^{-1}X^\top y
$$

### 4.2. 비선형 최소제곱 (Non-linear Least Squares)
- $\mathcal L_\infty$를 함께 추정해야 하는 손실 피팅처럼 파라미터가 비선형으로 얽혀 있으면 닫힌 해가 없다.
- 이 경우 반복적 개선(iterative refinement, 예: Levenberg–Marquardt류 방법)으로 $S=\sum_i(y_i-f(x_i))^2$를 최소화하는 파라미터를 찾는다.

## 5. 한 줄 요약
> 스케일링 법칙은 "작은 모델들의 로그-로그 곡선을 최소제곱으로 피팅해, 아직 학습하지 않은 큰 모델의 최적 하이퍼파라미터와 예상 손실을 미리 추정하는" 방법론이다.
