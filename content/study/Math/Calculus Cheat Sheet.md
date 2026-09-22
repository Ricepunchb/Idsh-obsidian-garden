---
title: Calculus Cheat Sheet
alias: 미적분 공식 모음
publish: true
date: 2026-09-22
tags:
  - Math
---

# 1. Taylor series

**Taylor 급수**는 어떤 함수를 한 점에서의 정보(그 점에서의 함숫값과 도함수들)만으로 다항식으로 근사하는 방법이다.
$$f(x)=\sum^{\infty}_{n=0}\dfrac{f^{(n)}(a)}{n!}(x-a)^n$$
$a=0$을 기준으로 전개한 것을 특별히 **Maclaurin series**라 부른다.

>[!example] $e^x$의 Taylor 급수 ($a=0$)
>$$e^x=1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+\cdots=\sum^{\infty}_{n=0}\dfrac{x^n}{n!}$$
>수렴 반경이 무한대라 모든 실수 $x$에서 $e^x$로 수렴한다.

**차수를 하나씩 올려가며 근사하는 과정**
- 0차: $f(x)\approx f(0)$ — $f(0)$ 높이의 수평선
- 1차: $f(x)\approx f(0)+f'(0)x$ — 원점에서 $f$와 값·기울기가 일치하는 접선
- 2차: $f(x)\approx f(0)+f'(0)x+\dfrac{f''(0)}{2}x^2$ — 값·기울기·곡률까지 일치하는 포물선

항을 하나씩 추가할 때마다 도함수 정보가 하나씩 더 들어가면서, 다항식이 원점 근처에서 $f$에 점점 더 밀착한다.

# 2. 미분 공식

>[!attention] 기본 공식 (암기)
>$$\dfrac{d}{dx}x^n=nx^{n-1},\qquad \dfrac{d}{dx}e^x=e^x,\qquad \dfrac{d}{dx}\log x=\dfrac1x$$
>$$\dfrac{d}{dx}\sin x=\cos x,\qquad \dfrac{d}{dx}\cos x=-\sin x$$

>[!attention] 합성/곱/몫 규칙
>$$\dfrac{d}{dx}f(g(x))=f'(g(x))\cdot g'(x) \qquad\text{(chain rule)}$$
>$$\dfrac{d}{dx}f(x)g(x)=f'(x)g(x)+f(x)g'(x) \qquad\text{(product rule)}$$
>$$\dfrac{d}{dx}\dfrac{f(x)}{g(x)}=\dfrac{f'(x)g(x)-f(x)g'(x)}{g(x)^2} \qquad\text{(quotient rule)}$$

# 3. 자주 쓰는 등식들 (Handy equations)

- **조화수(harmonic number)** $H_n=\sum^n_{k=1}\dfrac1k\approx\ln n$ (큰 $n$에서). [[Probability Puzzles|coupon collector]] 등에서 자주 등장.
- **로그평균 부등식**: $\sqrt{xy}\leq\dfrac{x+y}{2}$ for $x,y>0$ ([[Probability Inequalities|AM-GM]]의 $n=2$ 특수 케이스)
- **자연상수 $e$의 극한 표현들**
$$e=\lim_{n\to\infty}\left(1+\dfrac1n\right)^n$$
$n$ 대신 $-n$을 대입하면
$$\dfrac1e=\lim_{n\to\infty}\left(1-\dfrac1n\right)^n$$
$n$ 대신 $n/x$를 대입하면
$$e^x=\lim_{n\to\infty}\left(1+\dfrac xn\right)^n$$
- **지수함수의 급수 표현**: $e^x=\sum^{\infty}_{n=0}\dfrac{x^n}{n!}$ (1번의 Taylor 급수와 동일한 결과)

# 한 줄 요약
> Taylor 급수는 "한 점에서의 정보로 함수 전체를 다항식으로 흉내내기"이고, $e^x$가 그 대표 예시다. 조화수와 $e$의 극한 표현은 확률 퍼즐·부등식 증명에서 계속 재사용되니 따로 외워두면 좋다.
