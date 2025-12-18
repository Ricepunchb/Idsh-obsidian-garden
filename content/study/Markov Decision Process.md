---
title: Markov Decision Process
alias: MDP
publish: true
tags:
  - AI
  - RL
date: 2024-07-05
---
# Markov Decision Process (MDP)

## 1. 개요
순차적인 의사결정 문제(Sequential Decision Making)를 수학적으로 모델링한 프레임워크다.
에이전트(Agent)가 환경(Environment)과 상호작용하며, 매 시점 상태(State)를 관측하고 행동(Action)을 취해 보상(Reward)을 최대화하는 과정을 다룬다.

> [!abstract] 핵심 전제: 마르코프 성질 (Markov Property)
> **"미래는 오직 현재에 의해 결정되며, 과거는 상관없다."**
> 현재 상태 $s_t$는 과거의 모든 역사(History) 정보를 함축하고 있다고 가정한다.
> $$P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0)$$

## 2. MDP의 5대 구성 요소 (Tuple)
MDP는 보통 5개의 튜플 $\mathcal{M} = \langle S, A, P, R, \gamma \rangle$로 정의된다.

### 2.1. 상태 집합 ($S$, State Space)
-   에이전트가 관측할 수 있는 모든 상황의 집합.
-   $s_t \in S$

### 2.2. 행동 집합 ($A$, Action Space)
-   에이전트가 취할 수 있는 모든 행동의 집합.
-   $a_t \in A$

### 2.3. 전이 확률 함수 ($P$, Transition Probability)
-   현재 상태 $s_t$에서 행동 $a_t$를 했을 때, 다음 상태 $s_{t+1}$로 넘어갈 확률. (환경의 모델)
-   $$P_{ss'}^a = P(s_{t+1} = s' | s_t = s, a_t = a)$$

### 2.4. 보상 함수 ($R$, Reward Function)
-   어떤 상태에서 행동을 취해 변화가 일어났을 때 받는 즉각적인 보상값.
-   초안의 정의: $r_t = R(s_t, a_t, s_{t+1})$ (전이까지 고려한 가장 일반적 형태)
-   일반적 정의: 보통 기댓값으로 표현하여 $R(s, a) = \mathbb{E}[R_t | s_t=s, a_t=a]$로 쓰기도 한다.

### 2.5. 할인율 ($\gamma$, Discount Factor)
-   $0 \le \gamma \le 1$
-   미래에 받을 보상의 가치를 현재 시점으로 환산할 때 사용하는 비율.
-   **수학적 이유**: 무한한 시간(Infinite Horizon) 동안의 보상 합이 발산하지 않고 수렴하게 만듦.
-   **직관적 이유**: 미래의 불확실성을 반영하며, 당장의 보상을 선호하는 성향을 모델링함.
## 3. 에이전트의 목표 (Objective)

### 3.1. 반환값 (Return, $G_t$)
에피소드가 끝날 때까지 받는 **할인된 누적 보상의 합(Discounted Sum of Rewards)**이다. (초안의 $R(\tau)$)
$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

### 3.2. 정책 (Policy, $\pi$)
-   각 상태에서 어떤 행동을 할지 결정하는 전략(함수).
-   **Deterministic**: $a = \pi(s)$
-   **Stochastic**: $\pi(a|s) = P(a_t=a | s_t=s)$

### 3.3. 목적 함수 (Objective Function)
최적의 정책 $\pi^*$를 찾는 것이 목표다. 즉, **기대 반환값(Expected Return)을 최대화**해야 한다.
$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} [G_0]
$$

## 4. 가치 함수 (Value Function)
목적을 달성하기 위해 "현재 상태가 얼마나 좋은지"를 평가하는 함수가 필요하다.

### 4.1. 상태 가치 함수 (State-Value Function, $V_\pi$)
상태 $s$에서 시작하여 정책 $\pi$를 따랐을 때 기대되는 반환값.
$$
V_\pi(s) = \mathbb{E}_\pi [G_t | s_t = s]
$$

### 4.2. 행동 가치 함수 (Action-Value Function, $Q_\pi$)
상태 $s$에서 행동 $a$를 하고, 그 이후에는 정책 $\pi$를 따랐을 때 기대되는 반환값.
$$
Q_\pi(s, a) = \mathbb{E}_\pi [G_t | s_t = s, a_t = a]
$$

## 5. 벨만 방정식 (Bellman Equation)
현재 상태의 가치와 다음 상태의 가치 사이의 재귀적(Recursive) 관계식이다. MDP를 푸는 핵심 열쇠다.
$$
V_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s', r | s, a) [r + \gamma V_\pi(s')]
$$
> "현재의 가치 = (즉각적 보상 + 감가된 미래의 가치)의 기댓값"

## 6. Planning vs Learning

MDP 문제를 푼다는 것은 최적 정책 $\pi^*$를 찾는 것이다. 이때 $P$와 $R$을 아느냐 모르느냐가 핵심이다.

### 6.1. Planning (Dynamic Programming)
-   **상황**: 전이 함수 $P(s'|s,a)$와 보상 함수 $R(s,a)$를 **모두 알고 있음 (Model-based)**.
-   **방법**: 환경에 대한 완벽한 지도가 있으므로, 시뮬레이션 없이 계산만으로 최적해를 구함.
-   **알고리즘**: Policy Iteration, Value Iteration.

### 6.2. Reinforcement Learning (RL)
-   **상황**: $P$와 $R$을 **모르는 상태 (Model-free)**. (초안의 마지막 문장)
-   **방법**: 에이전트가 직접 환경과 상호작용하며 얻은 데이터(경험) $(s_t, a_t, r_t, s_{t+1})$를 통해 함수를 추정함.
-   **알고리즘**: Monte Carlo, SARSA, Q-Learning.

## 7. 요약
> **"MDP는 강화학습이 풀고자 하는 문제(Problem) 자체를 정의한 수학적 명세서이고, 강화학습은 그 문제의 정답지(Model)가 없을 때 답을 찾아가는 풀이법(Method)이다."**

