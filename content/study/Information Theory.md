---
title: Information Theory
alias: 정보 이론
publish: true
date: 2026-09-22
tags:
  - Math
  - Optimization
---
# 1. 개요
엔트로피, KL divergence, cross entropy는 서로 다른 개념이 아니라 하나의 수식에서 파생된 삼형제에 가깝다. LLM에서는 다음 토큰 예측 loss(cross entropy), 분포 간 유사도(KL), 모델 예측의 "퍼짐 정도"(엔트로피)로 각각 등장한다.

# 2. Entropy

분포 $p$가 얼마나 "불확실"한지, 즉 얼마나 퍼져있는지를 나타낸다.
$$
H(p)=-\sum_{x\in\mathcal X}p(x)\log p(x)
$$
logits $x_i$가 주어졌을 때는 아래처럼도 쓸 수 있다(softmax 확률의 엔트로피를 logit space에서 직접 표현한 형태).
$$
H(p)=\log\sum_ie^{x_i}-\underbrace{\frac{\sum_ie^{x_i}x_i}{\sum_i e^{x_i}}}_{\mathbb E[x]}
$$

# 3. Cross Entropy

분포 $q$(모델의 예측)로 분포 $p$(진짜 분포)를 표현하는 데 드는 평균 코드 길이, 라고 이해할 수 있다.
$$
\operatorname{CE}(p,q)=-\mathbb E_p[\log q]=-\sum_{x\in\mathcal X}p(x)\log q(x)
$$

# 4. KL Divergence

두 분포 $p,q$가 얼마나 다른지를 재는 비대칭적인 거리(distance는 아님, $\operatorname{KL}(p||q)\neq\operatorname{KL}(q||p)$).
$$
\operatorname{KL}(p\mid\mid q)=\sum_{x\in\mathcal X}p(x)\big(\log p(x)-\log q(x)\big)
$$
관련 활용처([[Loss Functions]] 참고): VAE의 잠재 분포 정규화, [[Knowledge Distillation]]의 teacher-student 분포 매칭, PPO의 정책 변화 제약 등.

# 5. 셋의 관계: CE = KL + H

$p$와 $q$ 사이의 cross entropy는 KL divergence에 $p$의 (줄일 수 없는) 고유 엔트로피를 더한 것과 같다.
$$
\operatorname{CE}(p,q)=\operatorname{KL}(p\mid\mid q)+H(p)
$$

>[!attention] 증명
>$$
>\begin{aligned}
>\operatorname{KL}(p\mid\mid q)&=\sum_{x}p(x)\big(\log p(x)-\log q(x)\big)\\
>&=\sum_{x}p(x)\log p(x)-\sum_{x}p(x)\log q(x)\\
>&=-H(p)+\operatorname{CE}(p,q)
>\end{aligned}
>$$
>양변을 정리하면 $\operatorname{CE}(p,q)=\operatorname{KL}(p||q)+H(p)$.

직관: $p$가 정답 분포로 고정되어 있으면 $H(p)$는 상수이므로, cross entropy를 최소화하는 것과 KL divergence를 최소화하는 것은 정확히 같은 최적화 문제다.

# 6. LLM의 Cross-Entropy Loss

다음 토큰 예측에서 정답 분포는 1-hot이므로(정답 토큰 확률 1, 나머지 0), cross entropy loss는 정답 토큰의 negative log-likelihood와 같아진다.
$$
\mathcal L(x)=-\sum_{t=1}^T\log p(x_t\mid x_{<t})
$$
(1-hot 타깃일 때 KL divergence와도 동치다 — $H(p)=0$이기 때문.)

## 6.1. 구현

`F.cross_entropy()`를 쓸 경우 logits/labels를 직접 shift해서 넣어줘야 한다.
```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=pad_idx)
```
padding mask까지 직접 다루는 구현은 이렇게 생겼다.
```python
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
logprobs = F.log_softmax(shift_logits, dim=-1)
token_logprobs = logprobs.gather(index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
# loss mask 적용
masked_logprobs = -token_logprobs * mask.float()
return masked_logprobs.sum() / mask.sum()
```
`log_softmax`를 직접 구현할 때의 수치 안정성 문제(overflow/underflow)는 [[Numerical Stability]] 참고, softmax+CE의 gradient 유도는 [[Gradients and Backpropagation]] 참고.

# 7. 한 줄 요약
> 엔트로피는 분포 하나의 "불확실성", cross entropy는 잘못된 분포 $q$로 진짜 분포 $p$를 표현하는 비용, KL divergence는 그 둘의 차이($\operatorname{CE}-H$)다. LLM의 next-token loss는 정답이 1-hot이라는 특수 케이스의 cross entropy이자 KL divergence다.
