---
title: Probability Inequalities
alias: 확률 부등식
publish: true
date: 2026-09-22
tags:
  - Math
  - Probability
---

> [!info] Markov·Chebyshev 부등식은 여기 없음
> 두 부등식은 이미 [[Chapter 4. Random variable|Chapter 4]] 9번 섹션에 증명까지 정리돼 있으므로 중복 작성하지 않는다. 이 노트는 그 외에 자주 쓰이는 부등식들을 모은 것이다.

# 1. Jensen's inequality

기댓값이 볼록/오목 변환과 어떻게 상호작용하는지 설명한다.

>[!attention] Jensen's inequality
>$$\varphi \text{ convex} \implies \varphi(E[X])\leq E[\varphi(X)]$$
>$$\varphi \text{ concave} \implies \varphi(E[X])\geq E[\varphi(X)]$$

>[!tip] 외우는 법
>$\varphi(x)=x^2$(볼록)이고 $X\in\{-1,+1\}$이 각각 $1/2$ 확률이라 해보자.
>- $\varphi(E[X])=\varphi(0)=0$
>- $E[\varphi(X)]=E[X^2]=1$
>
>$0\leq1$이니 "볼록이면 $\varphi(E[X])\leq E[\varphi(X)]$" 방향이 맞다는 걸 바로 확인할 수 있다.

**볼록함수** ($\varphi''\geq0$): $x^2, x^4,\dots$(짝수 차수), $|x|$, $a^x\,(a>1)$, $1/x\,(x>0)$, $\max(x_1,x_2,\dots)$
**오목함수** ($\varphi''\leq0$): $\sqrt x$, $\log x$, $x^p\,(0<p<1)$, $\min(x_1,x_2,\dots)$
선형함수는 볼록이면서 동시에 오목이라 등호가 성립한다.

기억해둘 만한 형태: $\log$가 오목이므로
$$\log E[X] \geq E[\log X]$$

# 2. AM-GM inequality

산술평균은 항상 기하평균보다 크거나 같다.
$$\dfrac{a_1+\cdots+a_n}{n}\geq\sqrt[n]{a_1\cdots a_n}$$

>[!note] proof.
>양변에 $\log$를 취하면 우변은 $\dfrac1n\sum_i\log a_i$가 되는데, $\log$가 오목함수이므로 Jensen's inequality에 의해 $\dfrac1n\sum_i\log a_i\leq\log\left(\dfrac1n\sum_ia_i\right)$. 양변을 다시 지수화하면 원하는 부등식을 얻는다.

# 3. Cauchy-Schwarz inequality

$$E[XY]^2\leq E[X^2]E[Y^2]$$

합 형태로는 다음과 같이 쓸 수 있다.
$$\left(\sum^n_{i=1}x_iy_i\right)^2\leq\left(\sum^n_{i=1}x_i^2\right)\left(\sum^n_{i=1}y_i^2\right)$$

# 4. Union bound

$$P(A_1\cup A_2\cup\cdots\cup A_n)\leq P(A_1)+P(A_2)+\cdots+P(A_n)$$

[[Chapter 3. Elements of probability|Chapter 3]]의 포함배제 원리(Proposition 2, $P(E\cup F)=P(E)+P(F)-P(E\cap F)$)에서 교집합 항을 버리고 부등식으로 느슨하게 만든 것과 같다. 각 사건의 확률이 작을 때 "여러 사건 중 하나라도 일어날 확률"의 상한을 빠르게 잡는 용도로 자주 쓰인다.

# 한 줄 요약
> Jensen은 기댓값과 볼록/오목 변환의 순서를 바꿀 때, AM-GM은 Jensen에서 $\log$를 대입하면 바로 나오고, Cauchy-Schwarz는 두 확률변수의 곱의 기댓값을 각각의 제곱의 기댓값으로 위에서 누르고, union bound는 포함배제 원리를 느슨하게 만든 버전이다.
