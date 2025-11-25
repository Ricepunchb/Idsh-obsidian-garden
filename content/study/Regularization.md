---
title: Regularization
alias: 규제화
publish: true
tags:
  - ML
date: 2024-07-05
---


모델의 Overfitting을 방지하기 위해 모델의 복잡성을 줄이는 방식

# Weight Decay
학습중에 큰 파라미터에는 큰 페널티를 부여해 파라미터를 규제하는 방식
Loss function에 벌점항인 penalty를 추가하여 구현
  $$\text{Loss}(\theta, \lambda|X,Y) = \text{Error}(\theta
  |X,Y) + \lambda\text{Penalty}(\theta)$$
## L1 Regularization
$$\text{penalty}(\theta)=|\theta_1|+|\theta_2| + \cdots + |\theta_p|$$
규제가 강해지면 파라미터가 0이 될수도 있음
변수 선택이 가능함

## Lasso Regression
선형회귀 + L1
$$\displaystyle L=\sum^n_{i=1}(y_i-\beta_0-\sum^p_{i=1}\beta_ix_i)^2+\lambda\sum^p_{i=1}|\beta_i|$$
## L2 Regularization
$$\text{penalty}(\theta)=\theta_1^2+\theta_2^2+\cdots+\theta_p^2$$
규제가 강해질수록 0으로 접근하지만 파라미터가 진짜 0이 되지는 않음

## Ridge Regression
선형회귀 + L2
$$\displaystyle L=\sum^n_{i=1}(y_i-\beta_0-\sum^p_{i=1}\beta_ix_i)^2+\lambda\sum^p_{i=1}\beta^2_i$$

## Elastic Net
L1+L2
양쪽 성질을 다 갖고있음
  $$ L=\text{ERROR}(\theta|X,Y)+\lambda_1\sum^n_{i=1}|\theta_i|+\lambda_2\sum^n_{i=1}\theta^2_i $$ $$L=\text{ERROR}(\theta|X,Y)+\alpha\left(\lambda\sum^p_{i=1}|\theta_i|+{1\over2}(1-\lambda)\sum^p_{i=1}\theta^2_i\right) $$
# Dropout
- 훈련중에 노드를 임의로 삭제하여 출력이 일부 노드에 의존하는 현상을 방지
  ㄴ좀 더 안정적인 예측이 가능
  
- 매 순방향 계산에서 p의 확률로 노드를 삭제하고 각 출력을 $\dfrac{1}{1-p}$만큼 스케일링해서 계산

- 예측 할땐 모든 노드를 그대로 사용