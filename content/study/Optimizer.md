---
title: Optimizer
alias: 옵티마이저
publish: true
date: 2026-09-22
tags:
  - Optimization
  - Neural-Network
---
# 1. 개요
옵티마이저(optimizer)는 계산된 gradient를 가지고 **"파라미터를 어느 방향으로, 얼마나" 움직일지** 결정하는 알고리즘이다. 가장 단순한 형태는 바닐라 SGD([[Neural Network]] 참고)지만, 실전에서는 거의 항상 Adam 계열을 쓴다.
$$
\theta\leftarrow\theta-\eta g_t
$$

>[!tip] Optimizer state vs Learning rate schedule
> 어떤 하이퍼파라미터 조절 로직을 "LR 스케줄"로 볼지 "옵티마이저"로 볼지 헷갈릴 수 있는데, 기준은 이렇다.
> - 현재 step $t$**에만** 의존한다 → 아마 LR 스케줄 (warmup, cosine decay 등)
> - 파라미터별 **history**(과거 gradient들의 통계)가 필요하다 → 옵티마이저 (Adam의 모멘트 추정 등)

# 2. Adam

파라미터 텐서마다 다음 네 가지를 유지한다: 파라미터 자체 $\theta$, gradient $g$, 1차 모멘트(모멘텀) $m$, 2차 모멘트(분산) $v$.
$$
\theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}
$$

## 2.1. 1차 모멘트 (모멘텀)
gradient의 이동평균이다.
$$
m\leftarrow\beta_1 m+(1-\beta_1)g
$$
gradient가 계속 같은 방향을 가리켜왔다면 그 방향으로 더 확신을 갖고 움직인다 — 시간에 걸쳐 속도(velocity)를 누적해서 노이즈가 많은 gradient를 뚫고 나가게 도와준다.

## 2.2. 2차 모멘트 (분산)
gradient 제곱의 이동평균이다.
$$
v\leftarrow\beta_2 v+(1-\beta_2)g^2
$$
결과적으로 파라미터마다 사실상 다른 학습률을 갖게 된다 — 꾸준히 큰 gradient를 받는 파라미터는 더 작은 스텝을 밟는다. gradient들을 같은 스케일로 정규화하는 효과가 있다.

## 2.3. Bias Correction
학습 초반에는 $m,v$가 0으로 초기화되어 있어 이동평균이 실제 값보다 작게 편향되는데, 이를 보정한다. ($t$는 1부터 시작하는 스텝 인덱스)
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}
$$

$\beta_1,\beta_2$는 모멘트 추정을 얼마나 천천히 업데이트할지 조절하는 하이퍼파라미터이고, $m,v$는 모두 0으로 초기화된다. 결과적으로 파라미터당 메모리는 (파라미터 자체 + gradient + $m$ + $v$로) 약 4배가 된다.

# 3. AdamW

Adam에 **weight decay**를 추가해서 파라미터를 0쪽으로 당기는 정규화 항을 더한 버전이다.
$$
\theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}-\eta\lambda\theta
$$
실무에서는 weight decay를 Adam 업데이트 *이전에* 적용하는 게 더 낫다 — weight decay 항 자체가 현재 파라미터 값에만 의존하기 때문이다 (아래 코드 참고).

## 3.1. 파라미터 그룹
옵티마이저를 초기화할 때 모델 파라미터와 `lr`(스텝 크기)을 넘겨준다. `params`는 서로 다른 하이퍼파라미터를 가질 수 있는 파라미터 그룹을 만드는 데 쓰인다 (예: 레이어별 다른 학습률). 보통 bias나 LayerNorm 파라미터에는 weight decay를 걸지 않는다.
```python
torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0},
])
```
`defaults` 딕셔너리는 특정 그룹에서 명시하지 않은 하이퍼파라미터의 기본값 역할을 한다.

## 3.2. 구현
```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 < betas[0] < 1 or not 0 < betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:  # 파라미터 그룹마다
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:  # 그룹 내 파라미터마다
                if p.grad is None:
                    continue
                state = self.state[p]

                # 0으로 상태 초기화
                t = state.get("t", 0)
                m, v = state.get("m", torch.zeros_like(p.data)), state.get("v", torch.zeros_like(p.data))

                # weight decay
                p.data -= lr * weight_decay * p.data

                # Adam update
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1 ** (t + 1))
                v_hat = v / (1 - beta2 ** (t + 1))
                p.data -= lr * m_hat / (v_hat.sqrt() + eps)

                # 옵티마이저 상태 업데이트
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
```

# 4. Gradient Clipping

전체 파라미터의 gradient에 대한 global norm을 계산해서, 이 값이 임계치를 넘으면 모든 gradient를 같은 비율로 줄여 임계치 아래로 맞춘다. 개별 스텝이 통제 불가능할 정도로 커지는 것(loss spike)을 막아준다.

# 5. Learning Rate Warmup

학습 초반 몇 스텝 동안 학습률을 0에서 목표값까지 서서히 끌어올리는 기법이다. 초기 몇 개의 배치가 아직 잘 세팅되지 않은 옵티마이저 상태(특히 Adam의 모멘트 추정)에 과도한 영향을 주는 것(primacy effect)을 완화한다. 보통 warmup 이후엔 cosine decay 등으로 학습률을 서서히 낮춘다.

# 6. 한 줄 요약
> Adam은 gradient의 1차 모멘트(방향의 관성)와 2차 모멘트(파라미터별 스케일 정규화)를 함께 추적해서 SGD보다 안정적으로 수렴하고, AdamW는 여기에 weight decay를 업데이트 이전 단계로 분리해서 더한 것이다. 여기에 학습 초반의 warmup과 gradient clipping을 더하면 실전 LLM 학습에서 쓰는 표준 레시피가 된다.
