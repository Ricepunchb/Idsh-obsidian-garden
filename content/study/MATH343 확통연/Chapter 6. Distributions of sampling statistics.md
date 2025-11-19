---
title: Chapter 6

publish: true

tags:
  - Study
  - Statistics

date: 2024-07-05
---



>[!attention] sample
>If $X_1,X_2, ... , X_n$ are independent r.v. having common distribution F, then we say they constitute **sample** or random sample from distribution F
>

같은 분포에서 왔으니 항상 평균과 분산이 같다

# Sample Mean
Let $X_1,X_2, ... ,X_n$ are samples with $\mu, \sigma^2$. 
분포는 노말일수도, 포아송 분포일수도 있다.
>[!attention] Sample Mean
>$$\bar{X}=\dfrac{X_1+\cdots+X_n}{n}$$

$E[\bar{X}]=E\left[\dfrac{X1+\cdots+X_n}{n}\right]=\mu$
$V[\bar{X}]=V\left[\dfrac{X1+\cdots+X_n}{n}\right]=\dfrac{\sigma^2}{n}$

# Central Limit Theorem
>[!note] Central Limit Theorem
>Let $X_1,...,X_n$ be i.i.d. (independent, identically distributed) r.v. with mean $\mu$, Variance $\sigma^2$
>Then, for $n$ large, distribution of $X_1+\cdots+X_n$ is approximately $N(n\mu, n\sigma^2)$

이것의 의의는 $n$이 커지면 $Z$로 standard normalization이 가능하다는 것

위의 Sample Mean 하고 비교해보자
$$X_1+\cdots+X_n \sim N(n\mu, n\sigma^2) $$
$$E[\bar X]={1\over n}E[X_1+\cdots+X_n]=\mu$$
$$V[\bar X]={1\over n^2}V[X_1+\cdots+X_n]={\sigma^2\over n}$$
결론은 똑같다
# Sample Variance
>[!attention] Sample Variance
>The statistic $S^2$
>Let $X_i \text { are random sample }\forall i$ s.t. $X_i\sim N(\mu,\sigma^2)$ 
>$$S^2={\sum^n_{i=1}(X_i-\bar X)^2 \over n-1}$$

$$E[S^2]={1\over n-1}E\left[\sum^n_{i=1}(X_i-\bar X)^2\right]$$
$$\begin{align} \Rightarrow (n-1)E[S^2] &=E\left[\sum^n_{i=1}(X_i^2-2X_i\bar X+\bar X^2)\right] \\ &=E\left[\sum^n_{i=1}X_i^2-n\bar X^2\right] \\ &=n(\sigma^2+\mu^2)-n(\sigma^2/n+\mu^2) \\ &=(n-1)\sigma^2 \end{align}$$
$$\therefore E[S^2]=\sigma^2$$
# Sampling Distributions from Normal Population

## Multivariate Normal Distribution
교수님의 추가 항목
>[!attention] Gaussian Random Vector (Normal Random Vector)
>Let $X \text{ random vector } :=(X_1,X_2,...,X_n).$
>$X$ is **Gaussian random vector**, or **Normal random vector**
>$\Leftrightarrow X$ have **Multivariate normal distribution**, or **Multivariate Gaussian distribution**, or **Joint normal distribution**
>If $X$ can be expressed as $$X=AZ+\mu$$

여기서 $A=\begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1k} \\ a_{21} & a_{22} & \cdots & a_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nk}\end{pmatrix}$ $Z=\begin{pmatrix} Z_1 \\ Z_2 \\ \vdots \\ Z_k \end{pmatrix}$ $\mu=\begin{pmatrix}\mu_1 \\ \mu_2 \\ \vdots \\ \mu_n \end{pmatrix}$ 이고 $Z_i$는 모두 i.i.d.


>[!note] Proposition.
>Let $X$ be Gaussian random vector. Then,
>
>$E[X]=(E[X_1], E[X_2],...,E[X_n])^T = \mu = (\mu_1,\mu_2,...,\mu_n)$
>$V[X]=\Sigma \text{ (covariance matrix) }=AA^T$
>$\phi_X(t)=E[e^{tX}]=E[e^{t(AZ+\mu)}]$

>[!note] Proposition
>$\begin{pmatrix} X\\ Y \end{pmatrix}$ be normal r.v.,$Cov(X,Y)=0$
>$\Rightarrow X\perp Y$

>[!note] Proposition
>$X_1,...,X_n$ : i.i.d. and $\sim N(\mu,\sigma^2)$
>$\Rightarrow \bar{X} \perp \begin{matrix}X_1-\bar{X} \\ \vdots \\ X_n-\bar{X} \end{matrix}$

>[!note] Thm 6.5.1.
>$X_1,...,X_n$ : sample such that $\forall i : X_i\sim N(\mu,\sigma^2)$. Then,
>
>1. $\bar{X} \perp S^2$
>2. $\bar{X} \sim N(\mu,\dfrac{\sigma^2}{n})$
>3. $\dfrac{(n-1)S^2}{\sigma^2}\sim \chi^2_{n-1}$ This is Student's Theorem

>[!note] Cor 6.5.2.
>Let $X_1,...,X_n$ sample from normal population. Then,
>$${\bar{X}-\mu \over S/\sqrt{n}} \sim t_{n-1}$$

# Sampling from Finite population

For any $\epsilon >0$,
$P\left(\left|{X_1+X_2+\cdots+X_n \over n}-\mu\right| > \epsilon\right) \rightarrow 0$ as $n \rightarrow 0$
