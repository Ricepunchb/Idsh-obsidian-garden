---
title: Policy Gradient
alias: 정책 경사법
publish: true
date: 2026-09-22
tags:
  - RL
  - RLHF
---
# Policy Gradient

## 1. 개요 및 표기
LLM post-training 맥락에서 [[Markov Decision Process]]의 일반적인 RL 표기를 다음과 같이 구체화한다.
- 행동 $a_t\in\mathcal V$: $t$ 시점의 next-token
- 상태 $s_t$: $t$ 시점까지의 텍스트 prefix $(s_0,a_0,\dots,a_{t-1})$
- $a_t\sim\pi_\theta(\cdot\mid s_t)$: LM 정책
- $s_0\sim p_0$: 시작 분포에서 샘플링한 프롬프트
- $\tau$: trajectory (rollout, episode)
- $R(\tau)$: trajectory $\tau$에 대한 보상

>[!attention] MDP 표기와의 대응
> [[Markov Decision Process]]에서 목적함수가 할인된 반환값 $G_0$의 기댓값이었다면, LLM post-training에서는 보통 할인 없이 trajectory 전체의 보상 $R(\tau)$를 직접 쓴다. "행동"은 다음 토큰을 생성하는 것에 대응한다.

## 2. 목표
$$
\text{maximize }J(\theta)=\mathbb E_{\tau\sim\pi_\theta} [R(\tau)]
$$
경사 상승법(gradient ascent)으로 최적화한다.
$$
\theta_{k+1}=\theta_k+\alpha\nabla_\theta J(\theta_k)
$$

## 3. REINFORCE (Vanilla Policy Gradient)
목적함수의 gradient는 다음과 같이 쓸 수 있다.
$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t) R(\tau)\right]
$$
- 형태상 [[Instruction Tuning]]의 SFT 업데이트와 동일하지만, 데이터 $\tau$가 **정책 자신으로부터 샘플링**되고 gradient가 $R(\tau)$로 **가중**된다는 점이 다르다.
- $R(\tau)>0$이면 $\tau$에 등장한 각 토큰 $a_t$에 대해 $\log\pi_\theta(a_t\mid s_t)$를 키우는 방향, $R(\tau)<0$이면 반대 방향으로 이동한다.
- $R(\tau)$의 크기가 클수록 스텝도 커진다.

### 3.1. 유도 (log-derivative trick)
$$
\begin{align*}
\nabla_\theta J(\theta)&=\nabla_\theta\mathbb E_{\tau\sim\pi_\theta} [R(\tau)]\\
&=\nabla_\theta\sum_\tau P(\tau\mid\theta)\,R(\tau)\\
&=\sum_\tau\nabla_\theta P(\tau\mid\theta)\,R(\tau)\\
&=\sum_\tau P(\tau\mid\theta)\nabla_\theta\log P(\tau\mid\theta)\,R(\tau)&\text{log-derivative trick: }\nabla P=P\,\nabla \log P\\
&=\mathbb E_{\tau\sim\pi_\theta}\nabla_\theta\log P(\tau\mid\theta)\,R(\tau)\\
&=\mathbb E_{\tau\sim\pi_\theta}\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\, R(\tau)
\end{align*}
$$
- 마지막 줄로 넘어갈 때, trajectory 확률 $P(\tau\mid\theta)$이 전이 확률(환경, $\theta$에 무관)과 $\prod_t\pi_\theta(a_t\mid s_t)$의 곱이므로 $\log P(\tau\mid\theta)$를 미분하면 환경 부분은 사라지고 정책 항만 남는다.

### 3.2. 배치 추정과 `pg_loss`
실제로는 $N$개의 rollout $\tau^{(i)}$를 정책 $\pi_\theta$에서 샘플링해 gradient를 추정한다.
$$
\hat g=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\,R(\tau^{(i)})
$$
구현에서는 `pg_loss.backward()`가 위 $\hat g$와 같은 gradient를 만들어내도록 스칼라 손실을 구성한다.
$$
L(\theta)=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\,R(\tau^{(i)})
$$

>[!warning] `pg_loss`는 진짜 손실이 아니다
> $L(\theta)$의 값 자체는 정책이 얼마나 좋은지를 알려주지 않는다. 현재 정책에서 샘플링한 데이터로 매번 새로 구성되는, 올바른 gradient를 만들어내기 위한 장치일 뿐이며 고정된 목적함수가 아니다.

## 4. Baseline: 분산 감소
### 4.1. 문제
Vanilla policy gradient는 분산이 매우 큰 추정량이다. 예를 들어 쉬운 프롬프트라서 배치 내 모든 응답이 양의 보상을 받았다고 하자. baseline이 없으면 배치 내에서 상대적으로 나빴던 응답까지도 전부 강화(reinforce)되어 버린다. 학습 전체로 평균 내면 괜찮아지지만, 개별 업데이트는 매우 노이즈가 크다. 평균 보상 같은 baseline을 쓰면 평균 이하 응답은 강화되지 않는다.

### 4.2. Baseline을 뺀 gradient
$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t) (R(\tau)-b(s_t))\right]
$$
- $b(s_t)$가 상태 $s_t$에만 의존하고 $a_t$에는 의존하지 않는 한, 이 baseline은 gradient 추정에 편향(bias)을 도입하지 않는다.
- $b(s_t)$가 $R(\tau)$와 상관관계가 높을수록 $R(\tau)-b(s_t)$가 작아져 gradient의 분산이 줄어든다.
- 목적함수 자체는 바뀌지 않는다 — gradient를 **추정하는 방식**만 달라질 뿐이다.

### 4.3. 왜 편향이 생기지 않는가
빼는 항 $B$의 기댓값이 0임을 보이면 된다.
$$
B=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t)b(s_t)\right]
$$
$X_t=\nabla_\theta\log\pi_\theta(a_t\mid s_t)b(s_t)$라 하면, 기댓값을 합 안으로 넣고(before: trajectory별로 $t$에 대해 합한 뒤 trajectory에 대해 평균 → after: 각 $t$마다 trajectory에 대해 평균한 뒤 그 평균들을 합), $X_t$가 $(s_t,a_t)$에만 의존한다는 사실을 이용해 trajectory 전체에 대한 기댓값을 $(s_t,a_t)$에 대한 기댓값으로 축소할 수 있다.
$$
B=\sum_{t=0}^T\mathbb E_{s_t}\left[\mathbb E_{a_t\mid s_t} X_t\right]
$$
안쪽 기댓값을 전개하면
$$
\begin{align*}
\mathbb E_{a_t\mid s_t} X_t&= b(s_t)\,\mathbb E_{a_t\mid s_t}\nabla_\theta\log\pi_\theta (a_t\mid s_t)&\text{$b$는 $a_t$에 의존하지 않음}\\
&= b(s_t)\,\sum_{a_t}\pi_\theta(a_t\mid s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\\
&= b(s_t)\,\sum_{a_t}\nabla_\theta\pi_\theta(a_t\mid s_t)&\text{log-derivative trick}\\
&= b(s_t)\nabla_\theta\sum_{a_t}\pi_\theta(a_t\mid s_t)\\
&= b(s_t)\nabla_\theta 1&\text{모든 행동의 확률 합은 1}\\
&= 0
\end{align*}
$$
- 핵심은 $b(s_t)$가 $s_t$에만 의존하기 때문에 $\mathbb E_{a_t\mid s_t}$ 밖으로 꺼낼 수 있고, 그 결과 $\sum_a\pi(a\mid s_t)=1$이 상수이므로 미분값이 0이 된다는 것이다. 따라서 $B=0$이고 baseline을 빼도 편향이 생기지 않는다.

### 4.4. Value function baseline과 Advantage
[[PPO]]가 사용하는 대표적인 baseline은 partial sequence $s_t$가 주어졌을 때 기대 보상을 추정하는 $V_\psi(s_t)$다.
$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t\mid s_t) (R(\tau)-V_\psi(s_t))\right]
$$
- 각 토큰이 받는 신호: "이 지점에서 예상했던 것보다 최종 보상이 더 좋았는가, 나빴는가?" — 안 좋아 보이던 응답을 좋은 결과로 바꾼 토큰일수록 더 큰 credit을 받는다.
- **Advantage와의 연결**: $R(\tau)$는 Q함수 $Q^\pi(s_t,a_t)$의 한 샘플이고, $V_\psi(s_t)$는 가치함수 $V^\pi(s_t)$의 추정치다 ($V^\pi(s)=\sum_{a\sim\pi(\cdot\mid s)}Q^\pi(s,a)$). 따라서 $R(\tau)-V_\psi(s_t)$는 advantage $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$의 몬테카를로 추정치다 — 단일 샘플 $\tau$의 보상으로 $Q^\pi$를 추정하는 셈이다.

### 4.5. 다른 baseline들
baseline은 결국 모두 "현재 상태에서의 기대 반환값" $V^\pi(s_t)$를 추정하려는 시도다.
- 학습된 value function (위 4.4, PPO)
- **RLOO**: 프롬프트당 $G$개의 응답을 샘플링하고, 각 응답의 baseline으로 **나머지 응답들의 평균 보상**을 사용한다. [[GRPO]]와 매우 비슷하지만 $i$를 제외한 나머지만 평균한다는 점과, std로 정규화하지 않는다는 점이 다르다.
- **REINFORCE++**: 배치 내 평균 보상을 baseline으로 사용한다.

## 5. Off-policy Policy Gradient
### 5.1. 문제
on-policy 방식은 gradient step마다 현재 정책으로 새로 inference(rollout)를 해야 하는데, 정책이 한 스텝에 크게 바뀌지 않는데도 매번 다시 샘플링하는 것은 비효율적이다.

### 5.2. Importance Sampling
이전 정책 $\pi_{\theta_\text{old}}$에서 샘플링한 rollout으로 현재 정책 $\pi_\theta$를 최적화한다. $p$에 대한 기댓값을 $q$의 샘플로 재구성하는 일반적인 importance sampling 배경:
$$
\mathbb E_{x\sim p}\,f(x)=\mathbb E_{x\sim q}\,\left[\frac{p(x)}{q(x)}f(x)\right]
$$
- $x$가 $q$보다 $p$에서 더 자주 나올수록, 그 샘플의 가중치가 커진다 (반대도 마찬가지).

### 5.3. Surrogate Objective
$$
\mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_{t=1}^T\underbrace{\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}}_{r_t}\, R(\tau)\right]
$$
- $r_t=\pi_\theta/\pi_{\theta_\text{old}}$는 importance sampling 스타일의 재가중치다.
- 이 목적함수는 원래의 $\mathcal J(\theta)$와 **다른 목적함수**다! true off-policy 추정은 $\mathcal J(\theta)$를 importance sampling으로 그대로 다시 쓴 것인데, ratio들의 곱이라 분산이 극도로 크다.
$$
\mathcal J(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\prod_{t=1}^T\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\,R(\tau)\right]
$$
$\mathcal J^\text{surrogate}$는 이 곱을 시점별 합으로 대체한 것이다.

### 5.4. Off-policy Gradient
$$
\nabla_\theta \mathcal J^\text{surrogate}(\theta)=\mathbb E_{\tau\sim\pi_{\theta_\text{old}}}\left[\sum_t\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\, R(\tau)\right]
$$
($r_t$의 분자만 $\theta$에 의존하므로, log-derivative trick을 다시 적용하면 유도된다.) 배치 추정:
$$
\hat g_\text{off-policy}=\frac 1N\sum_{i=1}^N\sum_{t=1}^T\frac{\pi_\theta(a_t^{(i)}\mid s_t^{(i)})}{\pi_{\theta_\text{old}}(a_t^{(i)}\mid s_t^{(i)})}\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})R(\tau^{(i)})
$$

>[!tip] 다음 단계
> 이 surrogate objective의 $r_t$에 clipping을 추가하면 [[PPO]]가 되고, PPO에 value function 대신 그룹 상대 정규화 advantage를 쓰면 [[GRPO]]가 된다.

## 6. 한 줄 요약
> Policy gradient는 정책이 샘플링한 trajectory의 보상으로 log-probability의 gradient를 가중해 업데이트하는 방법이며, baseline은 이 gradient의 분산을 편향 없이 줄이고, off-policy·importance sampling은 매 스텝 새로 샘플링하지 않고도 이전 정책의 rollout을 재사용할 수 있게 해준다.
