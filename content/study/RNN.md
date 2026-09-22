---
title: RNN
alias: 순환 신경망
publish: true
date: 2026-09-22
tags:
  - Architecture
  - LLM
---
# RNN (Recurrent Neural Network)

## 1. 개요
순차 데이터(sequence)를 처리하기 위해, 매 시점마다 이전 시점의 정보를 담은 hidden state를 다음 시점으로 넘겨가며 처리하는 아키텍처다. [[Transformer]] 이전 시퀀스 모델링의 표준이었고, [[Mamba]]의 State Space Model은 이 RNN적 순차 처리 방식을 다시 계승한 구조로 볼 수 있다.

## 2. Vanilla RNN
매 시점 $t$마다 입력 $x_t\in\mathbb R^D$와 이전 hidden state $h_{t-1}\in\mathbb R^H$를 받아 새로운 hidden state $h_t$를 만든다 ($D$는 입력 크기, $H$는 hidden 크기).
$$
h_t=\tanh(W_{x}x_t+W_{h}h_{t-1}+b)\\
y_t=W_\text{out} h_t+b_\text{out}
$$
- 가중치 $W_x\in\mathbb R^{H\times D}$, $W_h\in\mathbb R^{H\times H}$는 모든 시점에서 **공유**된다.

### 2.1. 기울기 소실 (Vanishing Gradient) 문제
$z_t=W_xx_t+W_hh_{t-1}+b$라 하면
$$
\frac{\partial h_t}{\partial h_{t-1}}=\operatorname{diag}(\tanh'(z_t)) W_h
$$
- $\tanh'\in(0,1]$이므로 이 항은 항상 값을 줄이는 쪽으로만 작용한다.
- 역전파 과정에서 이 야코비안이 시점 수만큼 반복해서 곱해지므로, $W_h$의 성질에 따라 기울기가 지수적으로 소실(vanish)되거나 폭발(explode)한다. 이 때문에 vanilla RNN은 긴 시퀀스에서 먼 과거의 정보를 학습하기 어렵다.

## 3. LSTM
LSTM은 hidden state와 별도로 **cell state** $c_t$를 두고, forget/input/output 세 개의 게이트로 정보 흐름을 제어해 기울기 소실 문제를 완화한다. 매 시점의 입력은 이전 hidden state $h_{t-1}$, 이전 cell state $c_{t-1}$, 현재 입력 $x_t$다.

- **forget gate**: cell state에서 무엇을 지울지 결정
$$
f_t=\sigma(W_f\cdot [h_{t-1}, x_t]+b_f)
$$
- **input gate**: 새로 어떤 정보를 쓸지 결정
$$
i_t=\sigma(W_i\cdot[h_{t-1},x_t]+b_i)
$$
- **후보 cell state**
$$
\tilde c_t=\tanh(W_c\cdot[h_{t-1},x_t]+b_c)
$$
- **cell state 업데이트**: 옛 cell state $c_{t-1}$에서 일부를 잊고(forget), 새 후보 $\tilde c_t$에서 일부를 더한다
$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
$$
- **output gate**: 새 hidden state로 무엇을 내보낼지 결정
$$
o_t=\sigma(W_o\cdot[h_{t-1},x_t]+b_o)
$$
- **새 hidden state**
$$
h_t=o_t\odot\tanh(c_t)
$$

>[!attention] LSTM이 작동하는 이유
> $f_t$가 1 근처로 유지되는 한, 기울기는 여러 시점을 거슬러 올라가도 소실되지 않고 전달될 수 있다. cell state는 "고속도로"처럼 흐르며 원소별 곱셈·덧셈만 거칠 뿐, 행렬곱이나 비선형 함수를 통과하지 않는다 — 정보는 오직 게이트를 통해서만 더해지거나 지워진다.

- **cell state vs hidden state의 역할 분리**가 LSTM의 핵심이다.
  - cell state $c_t$: 장기 기억(memory)
  - hidden state $h_t=o_t\odot\tanh(c_t)$: cell state를 걸러낸(filtered) 현재 출력이며, 동시에 다음 시점 게이트들이 참조하는 "자기 자신에 대한 질의(query)" 역할도 한다 ($t+1$ 시점의 게이트는 $c_t$가 아니라 $h_t$로부터 계산된다).
  - vanilla RNN은 하나의 벡터 $h_t$가 장기 기억과 현재 출력을 동시에 떠맡아야 했는데, 이 둘의 역할을 분리한 것이 LSTM이 작동하는 이유다.
- **GRU**는 reset, update 두 개의 게이트만으로 이를 단순화한 버전이다.

## 4. vs Transformer
- RNN에서는 먼 과거 시점의 기울기가 현재 시점의 처리에 영향을 주려면 중간의 모든 hidden state를 거쳐 살아남아야 한다.
- [[Attention]]은 토큰 간에 거리와 무관한 $O(1)$ 직접 경로를 제공한다 — 각 토큰이 다른 모든 토큰을 곧바로 참조하며, 모든 쌍(pairwise) 관계를 한 번에 계산한다. RNN에서 토큰 1의 정보가 토큰 100에 도달하려면 그 사이 모든 hidden state를 통과해야 하는 것과 대조적이다. 이 차이가 long-range dependency 처리에서 attention이 RNN보다 유리한 핵심 이유다.

## 5. 한 줄 요약
> RNN은 hidden state를 순차적으로 전달하며 시퀀스를 처리하는 구조이고, LSTM은 cell state와 hidden state를 분리해 장기 기억과 현재 출력의 역할을 나눔으로써 기울기 소실을 완화했지만, 여전히 먼 거리의 정보는 모든 중간 시점을 거쳐야 한다는 한계는 [[Attention]]의 $O(1)$ 직접 경로와 대비된다.
