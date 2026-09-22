---
title: Gradient Flow through Sampling
alias: 샘플링을 통과하는 그래디언트
publish: true
date: 2026-09-22
tags:
  - Optimization
  - RL
  - Generative-Model
---
# 1. 개요
`argmax`나 범주형(categorical) 샘플링처럼 이산적(discrete)이고 미분 불가능한 연산을 모델 안에 넣고 싶을 때가 있다 (예: 토큰을 하나 뽑아서 다음 연산에 넘기는 경우). 문제는 이런 연산을 통과하면 gradient가 흐르지 못한다는 것이다. 이 노트는 이산적인 선택을 하면서도 gradient를 흘려보내는 두 가지 표준 기법을 다룬다.

# 2. Gumbel-Max Trick

logits $z_1,\dots,z_k$가 주어졌을 때, 이에 대응하는 범주형 분포에서 샘플링하는 방법:
1. 독립적인 노이즈 $g_1,\dots,g_k\sim\operatorname{Gumbel}(0,1)$을 뽑는다.
2. $(z_1+g_1,\dots,z_k+g_k)$의 argmax를 취한다.

즉 "노이즈를 섞은 뒤 argmax"가 곧 확률적 샘플링과 동치라는 신기한 성질이다. 하지만 argmax 자체는 여전히 미분 불가능하다.

# 3. Gumbel-Softmax

argmax를 softmax로 바꾼다.
$$
\text{softmax}\big((z_1+g_1,\dots,z_k+g_k)/\tau\big)
$$
softmax는 어디서나 미분 가능하다! 온도 $\tau$가 높으면 부드러운(smooth) gradient가, 낮으면 거의 이산적인(discrete) 샘플이 나온다.

>[!tip] 핵심은 "미분 가능하게 만드는 것"이 아니라 "확률적으로 만드는 것"
> plain softmax는 애초에 이미 미분 가능하다. Gumbel-softmax의 진짜 역할은 여기에 **확률성(stochasticity)**을 추가하는 것이다.
> - plain softmax: $y=\operatorname{softmax}(\alpha)$ — 결정론적(deterministic), 항상 같은 soft mixture를 내놓음
> - 진짜 범주형 샘플링: $y=\operatorname{one\_hot}(\operatorname{sample}(\operatorname{softmax}(\alpha)))$ — 확률적이지만 미분 불가능
> - Gumbel-softmax: $y=\operatorname{softmax}((\alpha+G)/\tau)$ — **확률적**이면서(Gumbel 노이즈) **근사적으로 이산적**이고(낮은 온도) **미분 가능**하다. 탐색(exploration)과 gradient를 동시에 얻는 셈이다.

# 4. Straight-Through Estimator (STE)

미분 불가능한 함수 $f$를 순전파에서는 그대로 적용하되, 역전파에서는 마치 항등함수(identity)였던 것처럼 취급하는 기법이다.
- **순전파**: $y=f(x)$ (미분 불가능한 함수를 실제로 적용)
- **역전파**: $\dfrac{\partial\mathcal L}{\partial x}=\dfrac{\partial\mathcal L}{\partial y}$ (upstream gradient를 그대로 downstream gradient로 흘려보냄)

$f(x)$가 감소함수라면 이 근사는 신호의 방향 자체가 틀려버리므로 잘못된 결과를 낸다. 하지만 STE는 보통 반올림(rounding), 양자화(quantization), 증가하는 계단 함수처럼 **단조 증가(monotonically increasing)**하는 연산에 적용되므로, gradient의 크기는 틀려도 방향은 맞는다.

# 5. 한 줄 요약
> Gumbel-max는 argmax + Gumbel 노이즈로 범주형 샘플링을 구현하는 트릭이고, Gumbel-softmax는 그 argmax를 softmax로 완화해 확률성은 유지하면서 미분 가능하게 만든 버전이다. Straight-through estimator는 순전파에서는 미분 불가능한 함수를 그대로 쓰고 역전파에서는 항등함수인 척 gradient를 그대로 통과시키는, 더 범용적이지만 거친 근사다.
