---
title: Neural Network
alias: 인공신경망

publish: true

tags:
  - Study
  - AI

date: 2024-07-05
---



인간은 5살짜리도 쉽게 하지만 컴퓨터는 못하는 유형의 문제가 있다
- 안면인식, 언어처리, 음성인식 ...
- 인공지능은 이를 어떻게 해결할 수 있을까?
이래서 인간의 뇌를 닮은 수학적 모델인 인공신경망을 도입함

# 1. Perceptron
$$y=\sigma\left(\mathbf{w}^T\mathbf{x}+c\right)$$
여기서 $\sigma$는 non linear 함수여야만 함. 보통 시그모이드 함수, ReLu 함수

# 2. Gradient Descent 경사 하강법
일반적 최적화 문제를 풀 때 사용. $\min L(w,c)$

1. 파라미터 초기 설정 : $w^{(0)}=w_0, c^{(0)}=c_0$
2. 파라미터 업데이트 : $w^{(k+1)}=w^{(k)}-\alpha\dfrac{\partial L}{\partial w}\quad c^{(k+1)}=c^{(k)}-\alpha\dfrac{\partial L}{\partial c}$ $\alpha$는 학습률 튜닝 파라미터
3. 수렴할때까지 반복

## Stochastic Gradient Descent 확률적 경사하강법
- 일반 경사하강법은 한 스텝마다 그라디언트 계산에 계산량을 많이 쓴다
- SGD는 하나의 데이터만을 사용해 경사를 계산해 계산량을 줄임
- 각 스텝은 부정확하지만 결국 비슷하게 수렴, 그래서 빠르다
- 부정확해서 파라미터 업데이트는 더 많이 할 수도 있음
$$\text{Ordinary }\nabla L=-\sum^n_{i=1}2x_i(y_i-wx_i-c)$$
$$\text{Stochastic }\nabla L=-2x_i(y_i-wx_i-c)$$

## Mini-batch
batch 란 훈련에 사용되는 데이터 셋 하나의 단위

일반적 방법에선 파라미터 업데이트 한 번에 전체 배치를 사용

SGD 에서는 데이터 하나로 파라미터 업데이트

Mini-batch에서는 전체 훈련집합을 여러 집합으로 분할, 파라미터를 미니배치 하나마다 업데이트

## Learning Rate 학습률, $\alpha$
얼마나 빨리 파라미터를 업데이트 할 지 결정
Hessian 의 크기가 학습률의 최대치
- 큰$\alpha$ : 빠르지만 불안정한 학습, 수렴 안할수도 있음
- 작은 $\alpha$ : 느리지만 안정적인 학습