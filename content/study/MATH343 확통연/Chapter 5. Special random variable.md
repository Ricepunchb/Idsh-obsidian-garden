---
title: Chapter 5

publish: true


date: 2024-07-05
tags:
  - Math
  - Probability
---


확률함수는 Sample space가 달라도 확률이 같을 수 있다. 예를 들어, 주사위에서 홀수가 나오면 1이라 하고 동전에서 앞면이 나오면 1이라 할 때, 두 경우 모두 $P\{X=1\}$ 은 0.5로 같다.

# 1. Bernoulli and binomial r.v.
$X$가 *Bernoulli* r.v. 라 하면, 이벤트는 "성공", "실패" 처럼 두 가지 경우밖에 없다. 성공을 $X=1$, 실패를 $X=0$이라 한다면,
$P\{ X=1 \} =p, \quad P \{ X=0 \} = 1-p$ 이라 할 수 있다.

이 경우, $E[X]=1\cdot P\{X=1\}+0\cdot P\{X=0\}=p$ 를 항상 만족.
분산의 경우, $V[X]=E[X^2]-(E[X])^2=p-p^2=p(1-p)$ 이다.

$Y=X_1+X_2+\cdots X_n$이라 하고 각각의 $X_i$는 $i$번째 시행이고 서로 *independent* 할 때,
$$P\{Y=i\}=\binom{n}{i}p^i(1-p)^{n-i}$$
$$E[Y]=E[X_1+X_2+\cdots+X_n]=\sum^n_{i=1}E[X_i]=np $$
$$V[Y]=E[Y^2]-(E[Y])^2=V[X_1+X_2+\cdots+X_n]=\sum^n_{i=1}V[X_i]=np(1-p) $$
$Y\sim B(n,p)$  라 표기하고 $Y$가 *binomial* r.v.라 한다.

# 2. Poisson r.v.
>[!attention] Definition
discrete r.v.에 대해, 
$X$ 의 값이 $0, 1, 2,...$ 이면 *Possion random variable* 이다.
$$P\{X=i\}=e^{-\lambda}\dfrac{\lambda^i}{i!} \text{ with }\lambda>0$$

$E[X]=\sum_{i=0}iP\{X=i\}=\sum_{i=0}ie^{-\lambda}\dfrac{\lambda^i}{i!}=\lambda$
## Moment  generating function of Poisson
$$\begin{align} \phi(t)&=E[e^{tX}] \\ &=\sum^{\infty}_{i=0}e^{ti}e^{-\lambda}\lambda^i/i! \\ &= e^{-\lambda}e^{\lambda e^t} \\ &=e^{\lambda(e^t-1)} \end{align}$$
$$\phi'(t)=\lambda e^te^{\lambda(e^t-1)}$$
$$\phi''(t)=\lambda e^te^{\lambda(e^t-1)}+\lambda^2e^{2t}e^{\lambda(e^t-1)}$$
$t=0$ 에서,
$E[X]=\phi'(0)=\lambda$
$V[X]=\phi''(0)-(E[X])^2 = \lambda$
평균과 분산이 $\lambda$로 같은것은 포아송 분포의 특징!

$X_1, X_2$가 포아송 분포이고 각각 $\lambda_1, \lambda_2$를 가지면, $Y=X_1+X_2$ 에 대해, $Y$는 $\lambda_1+\lambda_2$를 가진다. 

$X\sim Y \leftrightarrow \phi_X(t)=\phi_Y(t)$.
포아송 분포의 distribution이 같으면 m.g.f도 같다.

### 이항분포의 극한으로서의 포아송 분포
포아송 분포는 $n$이 아주 크고 $p$가 아주 작은 이항분포의 극한으로도 유도된다. 고정된 구간을 $n$개의 아주 짧은 구간으로 쪼개서(구간마다 사건이 최대 1번만 일어난다고 가정) $np=\lambda$로 고정한 채 $n\to\infty$를 보내면,
$$\begin{align} P\{X=i\}&=\binom{n}{i}p^i(1-p)^{n-i} \\ &= \dfrac{n!}{i!(n-i)!}\left(\dfrac{\lambda}{n}\right)^i\left(1-\dfrac{\lambda}{n}\right)^{n-i} \\ &= \dfrac{\lambda^i}{i!}\cdot\underbrace{\dfrac{n!}{(n-i)!n^i}}_{\to 1}\cdot\underbrace{\left(1-\dfrac{\lambda}{n}\right)^{n}}_{\to e^{-\lambda}}\cdot\underbrace{\left(1-\dfrac\lambda n\right)^{-i}}_{\to 1} \\ &\longrightarrow e^{-\lambda}\dfrac{\lambda^i}{i!}\end{align}$$
그래서 포아송 분포는 "시행 횟수는 많지만 성공 확률은 낮은" 사건(하루 동안 걸려오는 전화 수, 특정 페이지의 오타 개수 등)을 모델링할 때 자연스럽게 등장한다.

# 3. Hypergeometric r.v.
멀쩡한 $N$개와 불량품 $M$개 중 $n$개를 뽑을때 멀쩡한거 $i$개를 뽑을 확률

$$P\{X=i\}=\dfrac{\binom{N}{i}\binom{M}{n-i}}{\binom{N+M}{n}},\quad i=0,1,\dots,\min(N,n)$$

$$X_i=\begin{cases}1 & \text{if selection is acceptable } \\ 0 & \text{else} \end{cases}$$
$\text{Let }X=X_1+X_2+\cdots+X_n$

1. $P\{X_1=1\}=\dfrac{N}{M+N},\, P\{X_1=0\}=\dfrac{M}{M+N}$

2. $\begin{align}P\{X_2=1\}  &= P\{X_2=1|X_1=1\}P\{X_1=1\}+P\{X_2=1|X_1=0\}P\{X_1=0\} \\ &= \dfrac{N-1}{M+N-1}\dfrac{N}{M+N}+\dfrac{N}{M+N-1}\dfrac{M}{M+N} \\ &= \dfrac{N}{M+N}\end{align}$
   By this, 
   $\begin{align} P\{X_k=1\} &:=\dfrac{\binom{N}{1}\cdot G}{G\cdot\left(\binom{N}{1}+\binom{M}{1}\right)}\quad \text{k번째가 N개중 1개고, 나머지 N-1개와 M개로 만들 수 있는 경우가 G개라 하자.}\\ &=\dfrac{N}{M+N} \end{align}$
   모든 시행에서 확률이 같지만, 그렇다고 각 시행이 independent 한 관계는 절대 아님! 
1. $P\{X_i=1, X_j=1\} = P\{X_i=1|X_j=1\}P\{X_j=1\}=\dfrac{N-1}{N+M-1}\dfrac{N}{N+M}$

2. $E[X_i]=0\cdot P\{X_i=0\}+1\cdot P\{X_i=1\}=\dfrac{N}{M+N}$

3. $E[X]=\sum^n_{i=1}E[X_i]=n\cdot\dfrac{N}{M+N}$

4. $V[X_i]=E[X_i^2]-(E[X_i])^2=\dfrac{N}{N+M}-\dfrac{N^2}{M^2+2NM+M^2}=\dfrac{NM}{(N+M)^2}$

5.  $\begin{align} Cov(X_i,X_j)&=E[X_iX_j]-E[X_i]E[X_j]\\ &= 1\cdot P{X_i=1, X_j=1}-E[X_i]E[X_j] \\ &= \dfrac{(N-1)(N+M-1)}{N(N+M)}-\dfrac{N^2}{(N+M)^2} \end{align}$ 

6. $\begin{align} V[X] &=\sum^n_{i=1}+2\sum\limits_{1\le i<j\le n}Cov(X_i,X_j) \\ &= \dfrac{nNM}{(N+M)^2} +2\sum\limits_{1\le i<j\le n}\left(\dfrac{(N-1)(N+M-1)}{N(N+M)}-\dfrac{N^2}{(N+M)^2}\right) \\ &= \dfrac{nNM}{(N+M)^2}+ \dfrac{n(n-1)}{2}\dfrac{-MN}{(N+M-1)(N+M)^2} \\ &= \dfrac{nNM}{(N+M)^2}\left(1-\dfrac{n-1}{N+M-1}\right)\end{align}$

# 4. Unifrom r.v.

>[!attention] Definition.
Random variable $X$는 $[\alpha,\beta]$에서 uniformly distributed.
$$\Leftrightarrow \text{p.d.f. of X is given by } f(x)= \begin{cases}\dfrac{1}{b-a} & a\le x\le b \\ 0 & \text{else} \end{cases}$$

1. $\displaystyle E[X]=\int_{-\infty}^{\infty}xf(x)dx = \int_{-a}^{b}{x\over b-a}dx = {a+b \over 2}$
2. $\displaystyle P\{\alpha\le x \le\beta\} = \int^{\beta}_{\alpha}f(x)dx=\int^{\beta}_{\alpha}{1\over b-a}dx = {\beta-\alpha\over b-a}$
3. $\displaystyle E[X^2]=\int_{-\infty}^{\infty}x^2f(x)dx = \int_{a}^{b}{x^2\over b-a}dx = {a^2+ab+b^2\over3}$
4. $\displaystyle V[X]=E[X^2]-(E[X])^2 = {a^2+ab+b^2\over3}-\left({a+b \over 2}\right)^2 = {(b-a)^2\over12}$

>[!example] Example e.
Random vector $X, Y$ 는 two-dimensional region $R$ 에 대해 uniform distribution을 갖고있다.
$$\Leftrightarrow \text{joint density function }f(x)=\begin{cases}c & (x,y)\in R \\ 0 & \text{else}\end{cases}$$
- $\displaystyle1=\int_Rf(x,y)dxdy=\int_R cdxdy=c\cdot(\text{area of R})$
- Let $A\subset R$. Then, $\displaystyle P\{(X,Y)\in A\}=\iint_{(x,y)\in A}f(x,y)dxdy=\iint_{(x,y)\in A}cdxdy=c\cdot\text{(Area of A)}={\text{Area of A}\over \text{Area of R}}$

# 5. Normal Distribution
>[!attention] Definition
$X\sim N(\mu.\sigma^2)$, $X$ is normally distributed with $\mu, \sigma^2$
$\Leftrightarrow \text{density function } f(x)=\dfrac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2\over 2\sigma^2}$

1. $\mu$ 에 대해 Symmetric 하다
2. $E[X]=\mu$
3. $V[X]=\sigma^2$

4. $X\sim N(\mu,\sigma^2),\, Z\sim N(0,1^2) \quad\Rightarrow\quad P\{{X-\mu\over\sigma}\le x\}=P\{Z\le x\}$

5. $X_1\perp X_2,\,X_1\sim N(\mu_1,\sigma_1^2),\,X_2\sim N(\mu_2,\sigma_2^2) \Rightarrow\quad X_1+X_2\sim N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$

6. Let $Y=X_1+X_2+\cdots+X_n,\,X_i\sim N(\mu_i,\sigma^2_i),\,\forall i,j : X_i\perp X_j$
   $\Rightarrow \mu=\mu_1+\mu_2+\cdots+\mu_n,\, \sigma^2=\sigma_1^2+\sigma_2^2+\cdots+\sigma_n^2,\,Y\sim N(\mu, \sigma^2)$
   
7. $\displaystyle \phi_Z(t)=E[e^{tZ}]=\int e^{ts}{1\over\sqrt{2\pi}}e^{-s^2/2}ds = {1\over\sqrt{2\pi}}\int e^{{-s^2\over2} + ts}ds = e^{t^2\over2}{1\over\sqrt{2\pi}}\int e^{-(s-t)^2\over2}ds = e^{t^2\over2}$

8. $\displaystyle \phi_X(t)=E[e^{tX}]=E[e^{t\mu}e^{t\sigma Z}] = e^{t\mu}E[e^{t\sigma Z}] =e^{\mu t+{\sigma^2t^2\over2}}$ 

9. 각 $X_i$는 전부 independent라 가정하면, 
   $\displaystyle E[e^{tY}]=E[e^{tX_1}e^{tX_2}\cdots e^{tX_n}]=\prod^n_{i=1}\mu_i$

## Definition of $\Phi$

 $$\Phi(x) = P\{Z\le x\}$$
 1. $\Phi(a) = P\{Z\le a\}=P\{Z\ge -a\}=1-P\{Z\le -a\}=1-\Phi(-a)$

## Definition of $z_\alpha$
$$z_\alpha \text{ is constant such that } P\{z_\alpha<Z\}=\alpha$$
$\Rightarrow \Phi(z_\alpha)=P\{Z\le z_\alpha\}=1-P\{Z> z_\alpha\}=1-\alpha$

# 6. Exponential r.v.

>[!attention] Definition
$X$ is continuous r.v. **exponential random variable** or exponentially distributed with parameter $\lambda$
$$\Leftrightarrow \text{p.d.f }f(x)=\begin{cases}\lambda e^{-\lambda x} & x\ge 0 \\ 0 & \text{else} \end{cases}$$

1. Cumulative distribution function $F(x)$
   $\displaystyle F(x)=P\{X\le x\}=\int^x_{-\infty}\lambda e^{-\lambda s}ds=1-e^{-\lambda x}$

2. $\phi(t)=E[e^{tx}]=\int e^{tx}f(x)dx=\int_0 e^{tx}\lambda e^{-\lambda x}dx=\lambda \int_0e^{(t-\lambda)x}dx = \left[{\lambda\over t-\lambda}e^{(t-\lambda)x}\right]^\infty_0$
   $={\lambda \over \lambda-t} \text{ for }t<\lambda$

3. $\phi'(t) = {\lambda \over (\lambda-t)^2}\quad \phi'(0) = {1 \over \lambda}$

4. $E[X]={1\over\lambda}$

5. $\phi''(t)={2\lambda\over(\lambda-t)^3}\quad\phi''(0)={2\over\lambda^2}=E[X^2]$

6. $V[X]={1\over\lambda^2}$

## Memoryless 성질
$$P\{X>(s+t)|X>t\}=P\{X>s\} \text{ for }s,t\ge0$$

>[!note] proof.
$$\text{LHS}={P\{X>s+t\}\over P\{X>t\}}={e^{-\lambda(s+t)}\over e^{-\lambda t}}=e^{-\lambda s}=\text{RHS}$$

형광등은 몇 년 쓴거랑 새거랑 기대수명이 똑같다

### 독립인 지수분포의 최솟값
$X_1\sim\text{Exp}(\lambda_1),\,X_2\sim\text{Exp}(\lambda_2)$ 이고 $X_1\perp X_2$ 이면,
$$\min(X_1,X_2)\sim\text{Exp}(\lambda_1+\lambda_2)$$
>[!note] proof.
>$$P\{\min(X_1,X_2)>t\}=P\{X_1>t\}P\{X_2>t\}=e^{-\lambda_1t}e^{-\lambda_2t}=e^{-(\lambda_1+\lambda_2)t}$$

예를 들어 이메일이 $\lambda_1$의 rate로, 문자가 $\lambda_2$의 rate로 온다면, "이메일이든 문자든 뭐든 먼저 오는 것"을 기다리는 시간은 $\text{Exp}(\lambda_1+\lambda_2)$를 따른다.

# 7. Geometric r.v.

>[!attention] Definition
$X$가 성공확률 $p$인 독립적인 베르누이 시행을 반복할 때, **첫 성공까지 필요한 시행 횟수**이면 $X$는 *geometric* r.v.이다.
$$P\{X=k\}=(1-p)^{k-1}p,\quad k=1,2,3,\dots$$

1. Tail probability는 실패만 $k$번 반복될 확률이므로 깔끔한 형태를 가진다.
   $$P\{X>k\}=(1-p)^k$$

2. $E[X]=\dfrac{1}{p}$, $V[X]=\dfrac{1-p}{p^2}$

>[!note] proof. (first-step analysis로 $E[X]$ 유도)
>첫 시행에서 확률 $p$로 성공하면 $X=1$, 확률 $1-p$로 실패하면 시행 1번을 소모하고 처음부터 다시 시작하는 것과 같으므로 실패 시 $X=1+E[X]$.
>$$E[X]=p\cdot1+(1-p)(1+E[X])=1+(1-p)E[X]$$
>정리하면 $pE[X]=1 \Rightarrow E[X]=1/p$.

### Memoryless 성질
Exponential r.v.와 마찬가지로, Geometric r.v.도 memoryless 하다.
$$P\{X>m+n \mid X>m\}=P\{X>n\}$$
이미 $m$번 실패했다는 사실이 앞으로 몇 번 더 실패할지에 아무 영향을 주지 않는다는 뜻이다. 실제로 **양의 정수 위에서 memoryless한 이산분포는 geometric 분포가 유일**하다 (반대로 $[0,\infty)$ 위에서는 exponential 분포가 유일).

### PMF의 합이 1임을 확인
등비급수(geometric series) $\displaystyle\sum^{\infty}_{k=0}r^k=\dfrac{1}{1-r}\;(|r|<1)$ 를 이용하면,
$$\sum^{\infty}_{k=1}(1-p)^{k-1}p = p\sum^{\infty}_{i=0}(1-p)^i = p\cdot\dfrac{1}{1-(1-p)}=p\cdot\dfrac{1}{p}=1$$

# 8. Distributions arising from the Normal
## A. Chi-Square Distribution
>[!attention] Definition
>$\chi_n^2$
>$Z_1,Z_2,...,Z_n \text{ are independent standard normal r.v.}$
>$X=Z_1^2+Z_2^2+\cdots+Z_n^2\Longleftrightarrow X\sim\chi^2_n$

p.d.f 는 교수님이 어려워서 생략

- $P\{X<0\}=0$

- $\chi_{\alpha,n}^2:=P\{X>\chi_{\alpha,n}^2\}=\alpha$

- $X\sim\chi^2_n,\,Y\sim\chi^2_m\quad\Rightarrow\quad X+Y\sim\chi^2_{n+m}$ for $X\perp Y$

- $E[X] = E[Z_1^2+...Z_n^2]=\sum^n_{i=1}E[Z_i^2]=\sum^n_{i=1}1=n$

- 분산은 증명해보세용 $V[X]=V[Z_1^2+...Z_n^2]=E[(Z_1^2+...Z_n^2)^2]-(E[Z_1^2+...Z_n^2])^2=E[(Z_1^2+...Z_n^2)^2]-n^2=2n$

## B. t-Distribution
>[!attention] Definition
>$$t_n={Z\over\sqrt{\dfrac{\chi^2_n}{n}}}={Z\over\sqrt{\dfrac{Z_1^2+...+Z_n^2}{n}}}$$ for every $Z_i$ are independent.

- $n$ 이 커질수록 $Z$에 가까워짐. 실제로 $n=5$만 되어도 비슷한 양상을 띔

- $0$을 중심으로 symmetric 함

- $$t_{\alpha,n}:=P\{T_n>t_{\alpha,n}\}=\alpha$$
- $E[T_n]=0$

- $E[T^2_n]=E\left[{nZ^2\over Z_1^2+...+Z_n^2}\right]=nE[Z^2]E\left[{1\over Z_1^2+...+Z_n^2}\right]={n\over n-2}$ for $n>2$

## C. F-Distribution
>[!attention] Definition
>$\displaystyle F_{n,m}={\chi_n^2/n \over \chi_m^2/m}$

- $$f_{\alpha,n,m}:=P\{F_{n,m}>f_{\alpha,n,m}\}=\alpha$$
- $E[F_{n,m}]=\dfrac{m}{m-2}$

- $V[F_{n,m}]=$ ??