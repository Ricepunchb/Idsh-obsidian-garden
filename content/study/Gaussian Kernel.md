---
title: Gaussian Kernel
alias: RBF Kernel
publish: true
date: 2025-12-15
---
# Gaussian Kernel (RBF Kernel)

## 1. 개요
**Radial Basis Function (RBF) Kernel**이라고도 불리며, 머신러닝(SVM, Kernel Regression)에서 가장 널리 쓰이는 커널이다.
[[NTK]] 이론을 이해하기 위한 선수 지식으로서, **"유한한 차원의 데이터를 어떻게 무한 차원의 특징 공간(Feature Space)으로 매핑하는가?"** 에 대한 답을 준다.

## 2. 수식적 정의
두 데이터 벡터 $x, x' \in \mathbb{R}^d$ 사이의 유사도를 측정한다.

$$
K(x, x') = \exp\left( -\gamma \|x - x'\|^2 \right) \quad \text{or} \quad \exp\left( -\frac{\|x - x'\|^2}{2\sigma^2} \right)
$$

-   **직관**: 유클리드 거리 $\|x - x'\|$가 가까울수록 1에 수렴하고, 멀어질수록 0에 수렴하는 **정규분포(Bell Curve) 모양**의 유사도 함수다.
-   $\gamma$ (Gamma): 분포의 뾰족함을 결정하는 파라미터. (클수록 뾰족함 $\to$ 근처 데이터만 봄).
## 3. 핵심: 무한 차원으로의 확장 (Proof)
이 부분이 NTK와 연결되는 가장 중요한 지점이다.
**"Gaussian Kernel을 계산한다는 것은, 데이터를 무한 차원으로 보낸 뒤 내적하는 것과 같다."**

### 3.1. 증명 (Taylor Expansion)
편의를 위해 1차원 데이터 $x, y$를 가정하고, $\gamma=1/2$로 두자.
$$
K(x, y) = \exp\left( -\frac{(x-y)^2}{2} \right)
$$
지수 법칙에 의해 항을 분리한다.
$$
= \exp\left( -\frac{x^2}{2} \right) \exp\left( -\frac{y^2}{2} \right) \exp(xy)
$$
여기서 $\exp(xy)$를 **테일러 급수(Taylor Series)** 로 전개한다 ($e^z = \sum_{n=0}^\infty \frac{z^n}{n!}$).
$$
= e^{-x^2/2} e^{-y^2/2} \sum_{n=0}^{\infty} \frac{(xy)^n}{n!}
$$
$$
= \sum_{n=0}^{\infty} \left( e^{-x^2/2} \frac{x^n}{\sqrt{n!}} \right) \left( e^{-y^2/2} \frac{y^n}{\sqrt{n!}} \right)
$$

### 3.2. 특징 맵 (Feature Map)의 발견
위 식은 두 무한 벡터의 내적(Dot Product) 형태 $\langle \phi(x), \phi(y) \rangle$와 같다.
즉, 매핑 함수 $\phi(x)$는 다음과 같이 정의된다.

$$
\phi(x) = e^{-x^2/2} \left[ 1, \frac{x}{\sqrt{1!}}, \frac{x^2}{\sqrt{2!}}, \frac{x^3}{\sqrt{3!}}, \dots \right]^T
$$

> [!abstract] 결론
> 우리가 단순히 $e^{-\|x-y\|^2}$라는 스칼라 값을 계산하는 행위는, 수학적으로 **$x$를 $1, x, x^2, x^3 \dots$의 모든 차수를 가진 무한 차원 공간으로 보낸 뒤 내적을 수행한 것**과 완벽하게 동일하다.

## 4. NTK와의 연결고리 (Connection)

| 구분                 | Gaussian Kernel                             | Neural Tangent Kernel (NTK)                             |
| :----------------- | :------------------------------------------ | :------------------------------------------------------ |
| **특징 맵 $\phi(x)$** | **고정됨 (Fixed)**<br/>($1, x, x^2, \dots$ 조합) | **네트워크 구조에 따라 결정됨**<br/>(Gradient $\nabla_\theta f(x)$) |
| **무한성의 기원**        | 테일러 급수의 무한 항                                | **무한한 뉴런 개수 (Width $\to \infty$)**                      |
| **역할**             | 데이터 간의 비선형 유사도 측정                           | 딥러닝 학습 과정의 수렴성 분석                                       |

NTK 이론은 **"신경망의 너비를 무한대로 늘리면, 그 신경망은 Gaussian Kernel처럼 '무한 차원 특징 공간'을 가진 커널 머신으로 수렴한다"** 는 것을 증명한 것이다.

## 5. Hyperparameter $\gamma$의 기하학적 의미
-   **Large $\gamma$ (High Frequency)**: 그래프가 뾰족하다. 데이터 포인트 하나하나를 개별적으로 기억한다. $\to$ **Overfitting** 위험 (RBF Network가 Nearest Neighbor처럼 동작).
-   **Small $\gamma$ (Low Frequency)**: 그래프가 완만하다. 멀리 있는 데이터까지 부드럽게 연결한다. $\to$ **Generalization** 유리.

> [!tip] NTK-Aware Scaling과의 관계
> "고주파수보다 저주파수를 먼저 학습한다"는 말은, 커널 관점에서 **"완만한(Small $\gamma$) 성분부터 피팅되고, 뾰족한(Large $\gamma$) 성분은 나중에 피팅된다"**는 뜻과 통한다.