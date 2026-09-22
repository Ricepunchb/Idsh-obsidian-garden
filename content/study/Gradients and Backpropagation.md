---
title: Gradients and Backpropagation
alias: 그래디언트와 역전파
publish: true
date: 2026-09-22
tags:
  - Optimization
  - Neural-Network
---
# 1. 개요
신경망 학습은 결국 "손실 $L$을 각 파라미터로 미분해서, 그 방향의 반대로 파라미터를 옮기는" 과정이다. 이 노트는 그 미분을 다변수/다층 함수에서 어떻게 체계적으로 계산하는지(Jacobian, chain rule) 정리하고, 실제 신경망에서 이를 효율적으로 계산하는 알고리즘인 역전파(Backpropagation)를 다룬다.

>[!abstract] 핵심 철학
> **"명시적으로 $\partial L/\partial \theta$의 전체 수식을 유도할 필요는 없다. 각 연산을 국소 gradient를 아는 단순한 노드로 쪼갠 뒤, chain rule로 이어붙이면 된다."**

# 2. Gradient, Jacobian, Hessian

미분(derivative)은 그 변수에 대한 함수의 "민감도"를 나타낸다. $\partial f/\partial x=3$이면 $x$를 작은 $h$만큼 바꿨을 때 $f(x)$는 대략 $3h$만큼 변한다.
$$
\frac{df(x)}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

- **Gradient** $\nabla f$: 다변수 함수 $f:\mathbb R^n\to\mathbb R$의 편미분들을 모은 벡터.
- **Jacobian**: 입력이 $n$개, 출력이 $m$개인 함수 $f(\mathbf x)=[f_1(x_1,...,x_n),...,f_m(x_1,...,x_n)]$에 대해, 모든 출력-입력 쌍의 편미분을 모은 $m\times n$ 행렬.
$$
\frac{\partial f}{\partial x}=\begin{bmatrix}\dfrac{\partial f_1}{\partial x_1}&\cdots &\dfrac{\partial f_1}{\partial x_n}\\
\vdots&\ddots&\vdots\\
\dfrac{\partial f_m}{\partial x_1}&\cdots&\dfrac{\partial f_m}{\partial x_n}\end{bmatrix}
$$
- **Hessian**: 입력이 $n$개, 출력이 스칼라인 함수의 2차 편미분 $n\times n$ 행렬, $H_{ij}=\partial^2f/\partial x_i\partial x_j$. 손실 함수의 Hessian은 손실 지형(loss landscape)의 곡률(curvature) 정보를 담고 있다.

# 3. Chain Rule

한 변수 함수의 합성이면 미분을 그냥 곱하면 된다.
$$
x=3y,\ y=x^2\ \Rightarrow\ \frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}=3\cdot 2x=6x
$$
여러 변수를 가진 함수의 합성이면 Jacobian끼리 곱한다.
$$
\mathbf h=f(\mathbf z),\ \mathbf z=\mathbf W\mathbf x+\mathbf b\ \Rightarrow\ \frac{\partial\mathbf h}{\partial\mathbf x}=\frac{\partial\mathbf h}{\partial\mathbf z}\frac{\partial\mathbf z}{\partial\mathbf x}
$$

## 3.1. 원소별(element-wise) 활성화 함수의 Jacobian

$\mathbf h=f(\mathbf z)$이고 $\mathbf h,\mathbf z\in\mathbb R^n$일 때, $h_i$는 $z_i$에만 의존하므로 Jacobian은 대각 행렬이다.
$$
\left(\frac{\partial \mathbf h}{\partial \mathbf z}\right)_{ij}=\frac{\partial h_i}{\partial z_j}=\begin{cases}f'(z_i)&i=j\\0&\text{otherwise}\end{cases}
\quad\Rightarrow\quad
\frac{\partial\mathbf h}{\partial \mathbf z}=\operatorname{diag}(f'(\mathbf z))
$$

## 3.2. 자주 쓰는 Jacobian 모음

$\mathbf z=\mathbf W\mathbf x+\mathbf b$, $\mathbf h=f(\mathbf z)$, $s=\mathbf u^\top\mathbf h$라는 전형적인 1-hidden-layer 신경망 setup에 대해:
$$
\frac{\partial \mathbf z}{\partial \mathbf x}=\mathbf W,\qquad
\frac{\partial \mathbf z}{\partial \mathbf b}=\mathbf I,\qquad
\frac{\partial \mathbf h}{\partial \mathbf z}=\operatorname{diag}(f'(\mathbf z)),\qquad
\frac{\partial s}{\partial \mathbf u}=\mathbf h^\top
$$
선형 레이어 $Z=XW+b$의 배치 버전 유도(shape 포함)는 [[신경망]] 참고.

## 3.3. 자주 쓰는 스칼라 미분

$$
\frac{d}{dx}\frac1x=-\frac1{x^2},\qquad
\frac{d}{dx}e^x=e^x,\qquad
\frac{d}{dx}\log x=\frac1x,\qquad
\frac{d}{dx}\sigma(x)=\sigma(x)(1-\sigma(x)),\qquad
\frac{d}{dx}\tanh(x)=1-\tanh^2(x)
$$

sigmoid의 미분은 quotient rule 대신 $(1+e^{-x})^{-1}$의 chain rule로 유도하는 게 더 깔끔하다.
$$
\frac{d}{dx}\sigma(x)=\frac{d}{dx}(1+e^{-x})^{-1}=-(1+e^{-x})^{-2}\cdot(-e^{-x})=\frac{e^{-x}}{(1+e^{-x})^2}=\sigma(x)(1-\sigma(x))
$$

Swish의 미분은 product rule + sigmoid 미분으로 유도된다.
$$
\begin{aligned}
\frac{\partial}{\partial x}\text{Swish}(x)&=\frac{\partial}{\partial x}\big[x\cdot\sigma(x)\big]=\sigma(x)+x\cdot\sigma'(x)\\
&=\sigma(x)+x\sigma(x)(1-\sigma(x))=\sigma(x)+\text{Swish}(x)(1-\sigma(x))
\end{aligned}
$$

# 4. Softmax + Cross-Entropy의 Gradient

LLM 학습에서 가장 많이 타는 경로이므로 직접 유도해볼 가치가 있다. logits $\mathbf z\in\mathbb R^{\mathcal V}$, softmax 확률 $\mathbf p\in\mathbb R^{\mathcal V}$, 정답 클래스 $t$.
$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad L=-\log p_t
$$

**CE loss의 $\mathbf p$에 대한 gradient**: $t$번째 항만 non-zero.
$$
\frac{\partial L}{\partial p_i}=\begin{cases}-\dfrac1{p_t}&i=t\\0&\text{otherwise}\end{cases}
$$

**softmax의 Jacobian** $\partial p_j/\partial z_i$:
$$
\frac{\partial p_j}{\partial z_i}=\begin{cases}p_j(1-p_j)&i=j\\-p_jp_i&i\neq j\end{cases}
$$

chain rule로 합치면, $\partial L/\partial p_i$가 $i=t$에서만 non-zero이므로 대부분의 항이 사라진다.
$$
\frac{\partial L}{\partial z_i}=\sum_j\frac{\partial L}{\partial p_j}\frac{\partial p_j}{\partial z_i}=-\frac1{p_t}\frac{\partial p_t}{\partial z_i}
$$
- 정답 토큰 ($i=t$): $\dfrac{\partial L}{\partial z_t}=-\dfrac1{p_t}\cdot p_t(1-p_t)=p_t-1$
- 나머지 토큰: $\dfrac{\partial L}{\partial z_i}=-\dfrac1{p_t}\cdot(-p_tp_i)=p_i$

즉,
$$
\frac{\partial L}{\partial \mathbf z}=\mathbf p-\operatorname{one\_hot}(t)
$$
**"예측 확률에서 정답 one-hot을 뺀 것"**이라는 극도로 깔끔한 결과가 나온다. 이게 바로 소프트맥스 층 바로 아래로 흘러들어가는 gradient다. ([[Loss Functions]]의 Categorical Cross Entropy 참고.)

# 5. Backpropagation

신경망의 순전파 수식은 **computation graph**(계산 그래프)로 표현할 수 있다. 어디까지를 하나의 "노드(gate)"로 묶을지는 편의의 문제이며, 보통 국소 gradient를 계산하기 쉬운 단위로 쪼갠다.

>[!tip] 직관
> 역전파는 각 gate가 "내 출력이 커져야 손실이 줄어드는지, 작아져야 줄어드는지, 얼마나 강하게"를 gradient 신호를 통해 서로 주고받는 과정으로 볼 수 있다. chain rule을 반복 적용해서, 각 gradient를 **upstream gradient(이미 계산됨) × local gradient**로 분해한다.

- 각 노드는 upstream gradient를 받아서 downstream gradient를 전달한다: **downstream = upstream × local**.
- $y$가 $a$와 $b$ 계산에 모두 쓰이는 갈래(branch)라면, gradient는 합쳐진다.
$$
\frac{\partial f}{\partial y}=\frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}
$$
- 노드별 직관: $+$는 upstream gradient를 각 항에 그대로 나눠주고(distribute), $\max$는 하나의 입력에만 라우팅하고(route), $\times$는 순전파의 계수를 서로 맞바꿔서(switch) 내려보낸다.

## 5.1. 알고리즘

1. 출력의 gradient를 1로 초기화한다.
2. 역위상정렬(reverse topological order)로 노드를 방문하며, 각 노드의 gradient를 그 노드의 successor들의 gradient로부터 계산한다.
3. 제대로 구현하면 순전파와 역전파의 big-$O$ 복잡도는 동일하다.

## 5.2. Automatic Differentiation

gradient 계산은 순전파의 symbolic한 표현으로부터 자동으로 유도될 수 있다. 각 노드 타입은 (a) 자신의 출력을 계산하는 법, (b) 출력에 대한 gradient가 주어졌을 때 입력에 대한 gradient를 계산하는 법을 알아야 한다 — 이 local gradient는 프레임워크를 구현하는 사람이 직접 작성해둔다.

## 5.3. Gradient Checking

모든 파라미터 $x$에 대해, $f(x-h)$와 $f(x+h)$를 직접 재계산해서 아래 근사식이 성립하는지 확인한다(디버깅용).
$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
$$

## 5.4. Activation Checkpointing

역전파에는 순전파에서 계산한 중간 activation들이 필요하므로, 신경망은 기본적으로 모든 중간 activation을 저장해둔다. **Activation(gradient) checkpointing**은 이 activation 중 일부("checkpoint")만 저장해서 메모리를 아끼고, 필요할 때 가장 가까운 checkpoint에서부터 부분적으로 다시 순전파를 돌려(recompute) 나머지 activation을 얻는 기법이다 — 메모리를 계산량과 맞바꾸는 것이다.

$N$개 레이어를 $K$개 세그먼트로 checkpoint한다면:
- 메모리: $O(N)\to O(K+N/K)$
- 역전파 계산량: $O(N)\to O(N+N(K-1)/K)$
- 최적 선택 $K=\sqrt N$일 때 메모리 $O(\sqrt N)$, 계산량 $\sim O(2N)$

## 5.5. 왜 스칼라에서 시작하는가

역전파는 모든 파라미터 $\theta$에 대해 $\partial L/\partial\theta$라는 "숫자 하나"를 계산하는 과정이므로, $L$이 스칼라일 때만 의미가 있다. PyTorch에서 스칼라에 대해 `.backward()`를 호출하면 암묵적으로 $\partial L/\partial L=1$로 시드(seed)된다.

토큰별 손실 $\ell_1,\dots,\ell_N$의 평균 $L=\frac1N\sum_i\ell_i$을 쓴다면, 미분의 선형성에 의해
$$
\frac{\partial L}{\partial\theta}=\frac1N\sum_{i=1}^N\frac{\partial \ell_i}{\partial\theta}
$$
즉 평균 손실의 gradient는 각 $\ell_i$를 개별적으로 역전파했을 때 나오는 gradient들의 평균과 정확히 같다.

## 5.6. Activation Gradient vs Parameter Gradient

upstream gradient는 항상 activation에 대한 것이고, 파라미터에 대한 gradient는 업데이트에 쓰이고 나면 거기서 멈춘다 — 파라미터는 computation graph의 leaf node이기 때문이다.
```
loss
  │  dL/dy2 (activation grad)
  ▼
Layer 2 ──→ dL/dW2 (param grad, 저장됨)
  │  dL/dy1 (activation grad)
  ▼
Layer 1 ──→ dL/dW1 (param grad, 저장됨)
  │  dL/dx  (activation grad — 보통 버려짐)
  ▼
input
```

# 6. 한 줄 요약
> Gradient는 함수의 민감도, Jacobian은 다변수 함수의 gradient 모음이고, chain rule은 이들을 곱해서 합성함수의 미분을 구하는 도구다. Backpropagation은 이 chain rule을 computation graph 위에서 역위상정렬 순서로 반복 적용해 모든 파라미터의 gradient를 순전파와 같은 복잡도로 계산하는 알고리즘이며, softmax+CE의 경우 그 결과가 $\mathbf p-\text{one\_hot}(t)$로 극도로 단순해진다.
