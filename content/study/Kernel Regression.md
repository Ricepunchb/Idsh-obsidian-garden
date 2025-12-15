---
title: Kernel Regression
publish: true
date: 2025-12-15
tags:
  - AI
---
# Kernel Regression (커널 회귀)

## 1. 개요
데이터의 분포를 특정 함수 형태(예: $y=ax+b$)로 가정하지 않고, **주어진 데이터 포인트들의 유사도(Kernel)를 기반으로** 새로운 값을 예측하는 **비모수적(Non-parametric)** 방법론이다.

> [!abstract] 핵심 직관: 가중 평균 (Weighted Average)
> **"새로운 점 $x$에서의 예측값은, $x$와 가까운 이웃 데이터들의 $y$값들을 가중 평균 낸 것이다."**
> 가까우면 가중치를 크게, 멀면 작게 준다.

## 2. Nadaraya-Watson Estimator (기본 형태)
통계학에서 가장 직관적으로 정의하는 커널 회귀의 형태다.

$$
\hat{y}(x) = \frac{\sum_{i=1}^{N} K(x, x_i) y_i}{\sum_{i=1}^{N} K(x, x_i)}
$$

-   $K(x, x_i)$: 커널 함수 (유사도 측정, 예: Gaussian).
-   **의미**: 모든 학습 데이터 $y_i$를 참조하되, $K$값이 큰(가까운) 데이터의 영향을 많이 받도록 정규화(Normalize)하여 합친다.

## 3. Kernel Ridge Regression (KRR)
머신러닝과 NTK 이론에서 주로 다루는 형태다. **선형 회귀(Ridge)** 를 **Dual Space**로 푼 것과 같다.

### 3.1. Primal Problem (일반 Ridge)
고차원 특징 공간 $\phi(x)$에서의 선형 회귀를 생각해보자.
$$
\min_{w} \sum_{i=1}^{N} (y_i - w^T \phi(x_i))^2 + \lambda \|w\|^2
$$
-   여기서 $w$를 구하려면 $\phi(x)$의 차원만큼 연산해야 한다. 만약 $\phi(x)$가 무한 차원이라면? 계산 불가능하다.

### 3.2. Dual Solution (Kernel Trick)
위 문제를 라그랑주 승수법으로 풀면, 최적의 가중치 $w^*$는 데이터들의 선형 결합으로 표현된다 ($w^* = \sum \alpha_i \phi(x_i)$). 이를 대입하면 **내적 연산**만 남게 되어 커널 함수로 대체할 수 있다.

최종 예측 함수 $f(x)$는 다음과 같다.
$$
f(x) = \mathbf{k}(x)^T (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}
$$

-   $\mathbf{K}$: Gram Matrix ($N \times N$). $\mathbf{K}_{ij} = K(x_i, x_j)$.
-   $\mathbf{k}(x)$: 새로운 입력 $x$와 학습 데이터 전체 간의 커널 벡터 ($N \times 1$).
-   $\mathbf{y}$: 학습 데이터의 타겟 벡터 ($N \times 1$).
-   $\lambda$: 정규화 계수 (Regularization).

> [!tip] 수식 해석
> $(\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}$ 부분은 **학습 데이터 전체에 대한 가중치 $\alpha$**를 미리 계산해 둔 것이다.
> 추론 시에는 새로운 $x$와 기존 데이터 간의 유사도 $\mathbf{k}(x)$를 구해서 이 $\alpha$와 내적한다.

## 4. NTK와의 연결고리

[[NTK|NTK 노트]]에서 **"무한 너비 신경망은 특징 맵 $\phi(x)$가 고정된다"** 고 했다.
즉, 학습 중에 $\phi(x)$가 변하지 않으므로, 신경망 학습은 결국 고정된 특징 공간에서의 선형 회귀, 즉 **Kernel Ridge Regression**을 푸는 것과 수학적으로 완전히 동일해진다.

-   **신경망의 커널**: $K_{NTK}(x, x') = \langle \nabla_\theta f(x), \nabla_\theta f(x') \rangle$
-   따라서 무한 너비 신경망의 수렴값은 위 **KRR 수식의 Closed-form Solution**으로 바로 구할 수 있다. (Gradient Descent를 무한 번 돌릴 필요 없이 한 방에 계산 가능)

## 5. Memory-based Learning
커널 회귀는 **Memory-based** (또는 Instance-based) 학습이다.
-   **Parametric (딥러닝)**: 학습 데이터를 압축해서 $w$에 저장하고 데이터는 버림.
-   **Non-parametric (커널 회귀)**: 학습 데이터($x_i, y_i$)를 **모두 메모리에 들고 있어야** 추론 가능.

## 6. 요약

| 구분 | Linear Regression | Kernel Regression (KRR) |
| :--- | :--- | :--- |
| **모델 형태** | $y = w^T x$ (직선/평면) | $y = \sum \alpha_i K(x, x_i)$ (곡선/초평면) |
| **파라미터** | $w$ (차원 크기) | $\alpha$ (데이터 개수 크기) |
| **복잡도** | 데이터가 많아도 모델 크기 일정 | **데이터가 많아지면 연산량 $O(N^3)$ 폭증** |
| **핵심** | 가중치 학습 | **유사도(Kernel) 정의** |