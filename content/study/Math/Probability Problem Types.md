---
title: Probability Problem Types
alias: 확률 문제 유형
publish: true
date: 2026-09-22
tags:
  - Math
  - Probability
---

확률 문제 중 특정 "패턴"으로 반복해서 등장하는 세 가지 풀이 기법을 모았다. [[Probability Puzzles]]와 [[Markov Chains]] 노트에서도 계속 재사용된다.

# 1. First-step analysis

>[!tip] 핵심 아이디어
> 과정의 **첫 한 스텝**에 집중해서, 구하고 싶은 기댓값을 "한 스텝 진행한 뒤 남은 문제"로 재귀적으로 표현한다. 그러면 자기 자신을 포함한 연립방정식이 나오고, 이를 풀면 답이 나온다.

**예제**: 공정한 동전을 던져서 앞면이 연속으로 2번 나올 때까지 걸리는 기대 횟수는?

- $E_0$ := 방금 뒷면이 나왔거나(또는 시작 상태)일 때 남은 기대 횟수
- $E_1$ := 방금 앞면이 나왔을 때 남은 기대 횟수

$$E_0=\underbrace{\tfrac12(1+E_0)}_{\text{뒷면, 진전 없음}}+\underbrace{\tfrac12(1+E_1)}_{\text{앞면}}$$
$$E_1=\underbrace{\tfrac12(1+E_0)}_{\text{뒷면, 처음부터}}+\underbrace{\tfrac12\cdot0}_{\text{앞면 2연속, 끝}}$$

두 식을 연립해서 풀면 $E_0=6$.

# 2. Counting in expectation (지시확률변수로 세기)

>[!tip] 핵심 아이디어
> "어떤 성질을 만족하는 항목이 몇 개나 있을까?" 같은 질문은, 각 항목마다 지시확률변수(indicator) $X_i=\mathbb 1[\text{항목 } i\text{가 성질을 만족}]$를 정의하고 $N=\sum_iX_i$로 놓은 뒤, **linearity of expectation**을 쓰면 항목들이 독립이 아니어도 깨끗하게 풀린다.
> $$E[N]=\sum_iE[X_i]=\sum_iP(X_i=1)$$

**예제**: 무작위 순열에서 고정점(자기 위치에 그대로 있는 원소)의 개수는 평균 몇 개일까? 대칭성에 의해 위치 $i$에 원래 $i$번째 원소가 올 확률은 $1/n$이므로, $E[N]=\sum^n_{i=1}\tfrac1n=1$.

# 3. Max/min of $n$개 확률변수

$X_1,\dots,X_n$이 iid이고 공통 CDF가 $F_X$일 때, 최댓값 $M=\max_iX_i$와 최솟값 $m=\min_iX_i$의 분포:

>[!attention] 최댓값의 CDF
>$$F_M(x)=P(M\leq x)=P(\text{모든 } X_i\leq x)=[F_X(x)]^n$$

>[!attention] 최솟값의 CDF
>$$P(m>x)=P(\text{모든 } X_i> x)=[1-F_X(x)]^n \;\Longrightarrow\; F_m(x)=1-[1-F_X(x)]^n$$

**기댓값을 구하는 두 가지 방법**
1. 3단계 레시피: CDF → (미분해서) PDF → 적분
2. Nonnegative RV라면 [[Chapter 4. Random variable|tail-sum formula]] $E[X]=\int_0^\infty P(X>x)dx$를 바로 적용

>[!example] $X_i\sim\text{Uniform}[0,1]$일 때
>- 최댓값: $F_M(x)=x^n \Rightarrow f_M(x)=nx^{n-1} \Rightarrow E[M]=\int_0^1x\cdot nx^{n-1}dx=\dfrac{n}{n+1}$
>  - 직관: $n$개의 균등한 점이 $[0,1]$ 구간을 $n+1$개의 조각으로 나누고, 대칭성에 의해 각 조각의 평균 길이는 $\tfrac{1}{n+1}$이어야 한다.
>- 최솟값: $E[m]=\int_0^1x\cdot n(1-x)^{n-1}dx=\dfrac{1}{n+1}$

>[!example] $X_i\sim\text{Exponential}(\lambda)$일 때
>tail-sum formula로 바로: $P(m>x)=e^{-n\lambda x} \Rightarrow E[m]=\int_0^\infty e^{-n\lambda x}dx=\dfrac{1}{n\lambda}$. 최댓값은 닫힌 형태가 없지만 $n$에 대해 로그 오더로 커진다는 점만 기억해두면 유용하다.

>[!example] $X_i\sim N(0,1)$일 때
>유도는 까다롭지만 $E[\max_iX_i]\approx\sqrt{2\ln n}$라는 근사식은 알아둘 만하다.

# 한 줄 요약
> 재귀적인 기댓값 문제는 first-step analysis, "몇 개나 있을까" 문제는 지시확률변수+linearity, min/max 문제는 CDF를 거듭제곱해서 구한다.
