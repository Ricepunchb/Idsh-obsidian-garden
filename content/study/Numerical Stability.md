---
title: Numerical Stability
alias: 수치 안정성
publish: true
date: 2026-09-22
tags:
  - Optimization
  - LLM
---
# 1. 개요
부동소수점(floating point) 연산에는 한계가 있다. 특히 지수함수와 로그함수를 다룰 때는 다음 세 가지를 항상 조심해야 한다.
- $\exp(x)$: $x$가 크면 $\infty$로 **overflow**
- $\log(x)$: $x$가 0에 가까우면 $-\infty$로 **underflow**
- $\log(x)$: $x$가 1에 가까우면 **정밀도(precision) 문제** ($\log(1)=0$이라 근처 값들의 차이가 뭉개짐)

LLM에서는 softmax/log-softmax, attention의 logsumexp 등 거의 모든 곳에서 이 문제를 만나게 된다.

>[!tip] 표준 해법
> $e^x$가 들어간 식을 수치적으로 안정시키는 표준 도구는, 어떤 큰 수 $m$(보통 $m=x_\text{max}$)에 대해 $e^{-m}$을 분자·분모에 함께 곱해서 무엇이 남는지 보는 것이다.

# 2. Softmax의 Overflow 방지

$$
\text{softmax}(x)_i=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$
$x_i$가 크면 $\exp(x_i)$가 바로 overflow된다. softmax가 모든 입력에 같은 상수를 빼도 값이 변하지 않는다는 성질을 이용한다.
$$
\begin{aligned}
\text{softmax}(x)_i &= \frac{e^{x_i}}{\sum_j e^{x_j}}
= \frac{e^{x_i-c}\cdot e^c}{\sum_j e^{x_j-c}\cdot e^c}
= \frac{e^{x_i-c}}{\sum_j e^{x_j-c}}
=\text{softmax}(x-c)_i
\end{aligned}
$$
$c=x_\text{max}$로 두면 모든 $x_i-c\le 0$이 되어 가장 큰 지수도 $e^0=1$이 되므로 overflow가 사라진다.
$$
\text{softmax}(x)_i=\frac{e^{x_i-x_\text{max}}}{\sum_j e^{x_j-x_\text{max}}}
$$

# 3. Log-Softmax

log-softmax를 `log(softmax(x))`처럼 곧이곧대로 계산하면, 확률이 낮은 클래스(0에 가까운 입력)의 로그를 취하는 과정이 불안정해진다. 대신 `x - logsumexp(x)`를 쓰면 작은 확률값을 직접 만들 일이 없어져 안정적이다.
$$
\log(\text{softmax}(x))_i = \log \frac{e^{x_i}}{\sum_j e^{x_j}} = x_i-\log\sum_j e^{x_j}
$$

# 4. `logsumexp`

$\log\sum_i e^{x_i}$가 불안정한 이유는 세 가지다: 큰 $x_i$에서 $\exp$가 overflow, 모든 $x_i$가 매우 작으면 $\log(0)\to-\infty$로 underflow, 합이 1 근처면 $\log(1)$ 근처의 정밀도 문제.

직관: $x_i$들을 상수만큼 빼서 작게 만든 뒤, 마지막에 그 상수를 다시 더해주면 된다.
$$
\begin{aligned}
\log\sum_i e^{x_i}&=\log\sum_i\left(e^{x_i-x_\text{max}} \cdot e^{x_\text{max}}\right)
= \log \left(e^{x_\text{max}}\sum_i e^{x_i-x_\text{max}}\right)\\
&=x_\text{max}+\log\sum_i e^{x_i-x_\text{max}}
\end{aligned}
$$
가장 큰 항이 $e^0=1$이 되어 overflow가 없고, 합이 최소 1 이상이라 $\log(0)$을 계산할 일도 없다.

# 5. Online (Streaming) Softmax

naive한 softmax는 $x_\text{max}$를 구하는 패스와 분모 $\sum_j e^{x_j-x_\text{max}}$를 구하는 패스, 총 두 번을 필요로 한다. **online softmax**는 이 둘을 한 패스로 융합(fuse)한다 — FlashAttention의 핵심 아이디어 중 하나다.

running maximum $m_k=\max(x_1,...,x_k)$과 running (shifted) denominator $d_k=\sum_j e^{x_j-m_k}$를 유지하다가, 새 값 $x_{k+1}$이 들어올 때마다 업데이트한다.
$$
m_{k+1}\leftarrow\max(m_k,x_{k+1}),\qquad
d_{k+1}\leftarrow d_k\cdot e^{m_k-m_{k+1}}+e^{x_{k+1}-m_{k+1}}
$$

>[!attention] $d_{k+1}$ 업데이트가 맞는 이유
>$$
>\begin{aligned}
>d_{k+1}&=\sum_{j=1}^{k+1}e^{x_j-m_{k+1}}\\
>&= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_{k+1}}&\text{마지막 항 분리}\\
>&= e^{x_{k+1}-m_{k+1}}+\sum_{j=1}^k e^{x_j-m_k}e^{m_k-m_{k+1}}&\text{대수적 정리}\\
>&= \underbrace{d_k\cdot e^{m_k-m_{k+1}}}_\text{기존 항들의 재조정}+\underbrace{e^{x_{k+1}-m_{k+1}}}_\text{새 항}
>\end{aligned}
>$$
>최댓값이 갱신되지 않으면 $m_k=m_{k+1}$이라 재조정 계수 $e^{m_k-m_{k+1}}=1$이 되어 그냥 새 항만 더해지는 형태로 자연스럽게 축약된다.

모든 값을 다 처리한 뒤, softmax 자체가 필요하면 $e^{x_i-m_S}/d_S$를 구하는 한 번의 추가 패스가 필요하다.

## 5.1. Weighted Sum까지 융합 (FlashAttention)

logits $x_i$와 value 벡터 $v_i$의 스트림이 주어졌을 때, softmax 가중합을 다음처럼도 쓸 수 있다.
$$
o=\sum_ip_iv_i=\frac{\sum_i e^{x_i}v_i}{\sum_i e^{x_i}}
$$
FlashAttention에서는 $x_i=q\cdot k_i$, $v_i$는 value 벡터이고, $o$가 query $q$에 대한 attention 출력이 된다. overflow를 막기 위해 분자·분모에 $e^{-x_\text{max}}$를 곱한 안정적인 버전을 쓴다.
$$
o=\frac{\sum_i e^{x_i-x_\text{max}}v_i}{\sum_i e^{x_i-x_\text{max}}}
$$
$m_k$, $d_k$에 더해 running numerator $o_k\in\mathbb R^H$($H$ = head dimension)를 함께 유지한다.
$$
o_k=\sum_{i=1}^ke^{x_i-m_k}v_i,\qquad
o_{k+1}=o_k\cdot e^{m_k-m_{k+1}}+e^{x_{k+1}-m_{k+1}}v_{k+1}
$$
($d_k$ 업데이트와 완전히 동일한 논리로 유도된다.) $N$개를 다 처리하면 $(m_N,d_N,o_N)$이 남고, 최종 attention 출력은 그냥 $o_N/d_N$이다.

# 6. 한 줄 요약
> $\exp$와 $\log$가 얽힌 식을 안정화하는 표준 트릭은 상수(보통 $x_\text{max}$)를 빼서 모든 지수를 0 이하로 만든 뒤 마지막에 다시 보정하는 것이다. 이 아이디어 하나로 안정적인 softmax, log-softmax, logsumexp를 모두 유도할 수 있고, 이를 한 패스로 스트리밍하며 값까지 가중합하도록 확장한 것이 FlashAttention의 online softmax다.
