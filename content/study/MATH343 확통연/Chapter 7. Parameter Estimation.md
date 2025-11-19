---
title: Chapter 7

publish: true

tags:
  - Study
  - Statistics

date: 2024-07-05
---


# Maximum Likelihood Estimator

**Estimator** : unknown parameter $\theta$를 *estimate* 하는데 사용되는 statistic. 계획, 공식과 비슷한 개념이다.
예). 
# Interval Estimates

$X_1,...,X_n$ : sample from normal population having unknown mean $\mu$, known variance $\sigma^2$
$\mu$는 모르고 $\sigma^2$만 알 때

$P\left(-Z_{\alpha/2}{\sigma\over\sqrt{n}} < \bar{X}-\mu <Z_{\alpha/2}{\sigma\over\sqrt{n}}\right)=1-\alpha$
$P\left(\bar{X}-Z_{\alpha/2}{\sigma\over\sqrt{n}} <\mu <\bar{X}+Z_{\alpha/2}{\sigma\over\sqrt{n}}\right)=1-\alpha$

>[!attention] Two-sided $100(1-\alpha)$% confidence interval for $\mu$
>$$\mu \in (\bar{x}-Z_{\alpha/2}{\sigma\over\sqrt{n}}, \bar{x}+Z_{\alpha/2}{\sigma\over\sqrt{n}})$$

>[!attention] One-sided ver.
>$P({\bar{X}-\mu \over \sigma/\sqrt{n}} < Z_\alpha) = 1-\alpha$
>$P(\bar{X}-\mu < Z_\alpha{\sigma\over\sqrt{n}}) = 1-\alpha$
>$P(\bar{X}- Z_\alpha{\sigma\over\sqrt{n}} < \mu) = 1-\alpha$
>$$\mu\in\left(\bar{x}-Z_\alpha{\sigma\over\sqrt{n}}, \infty\right)$$
>One-sided Lower : $$\mu\in\left(-\infty, \bar{x}+Z_\alpha{\sigma\over\sqrt{n}} \right)$$

>[!attention] Length of  Confidence Interval
>$$2Z_{\alpha/2}{\sigma\over\sqrt{n}}=0.1 \quad \Rightarrow \quad n\ge(20Z_{\alpha/2}\sigma)^2$$

## Confidence Interval for a Normal Mean when the Variance is Unknown
$X_1,...,X_n$ : i.i.d., $X_i\sim N(\mu,\sigma^2), \mu$ : unknown, $\sigma^2$ : unknown
$\mu,\sigma^2$ 둘 다 모를때
## Prediction Interval
$X_{n+1}$ 값의 예측 범위

$X_1,...,X_{n+1} \sim N(\mu,\sigma^2)$이면
$\bar{X}\sim N(\mu,{\sigma^2\over n}),\,-X_{n+1}\sim N(-\mu,\sigma^2)$
$\bar{X}\perp -X_{n+1} \Rightarrow \bar{X}-X_{n+1}\sim N(0,\sigma+{\sigma^2\over n})$
Recall that $\displaystyle\bar{X}\perp S^2,\quad{(n-1)S^2\over \sigma^2}\sim \chi^2_{n-1},\quad {Z\over \sqrt{\chi^2_n\over n}}\sim t_n$
$\Rightarrow\dfrac{\bar{X}-X_{n+1}}{\sigma\sqrt{1+1/n}}\Big/\sqrt{\dfrac{(n-1)S^2}{\sigma^2(n-1)}}\sim t_{n-1}$
$\displaystyle\Rightarrow\dfrac{\bar{X}-X_{n+1}}{S\sqrt{1+1/n}}\sim t_{n-1}$
$\Rightarrow X_{n+1}\in(\bar{x_n}-t_{\alpha/2,n-1}S_n\sqrt{1+1/n}, \bar{x_n}+t_{\alpha/2,n-1}S_n\sqrt{1+1/n})$

## Confidence Interval for the Variance of a Normal Distribution
$\mu, \sigma^2$을 둘 다 모를때
$S^2$을 $\sigma^2$의 estimator로 사용함

$\dfrac{(n-1)S^2}{\sigma^2}\sim\chi_{n-1} \Rightarrow P \{ \square < \dfrac{(n-1)S^2}{\sigma^2} < \chi^2_{\alpha/2,n-1} \}=1-\alpha$
$\square=\chi^2_{1-\alpha/2,n-1} \Rightarrow P\{\dfrac{(n-1)S^2}{\chi^2_{\alpha/2,n-1}} < \sigma^2 < \dfrac{(n-1)S^2}{\chi^2_{1-\alpha/2,n-1}}\} = 1-\alpha$
$\sigma^2\in\left( \dfrac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}, \dfrac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}} \right)$

# Estimating the Difference in Means of Two Normal Populations

$X_1,...,X_n$ : i.i.d. $X_i\sim N(\mu_1,\sigma_1^2)$
$Y_1,...,Y_m$ : i.i.d. $Y_i\sim N(\mu_2,\sigma_2^2)$ and $\sigma_1,\sigma_2$ 을 알 때
$\mu_1-\mu_2=?$

$\bar{X}={1\over n}\sum^n_{i=1}X_i,\quad\bar{Y}={1\over m}\sum^m_{i=1}Y_i$
$\Rightarrow \bar{X}-\bar{Y}\sim N(\mu_1-\mu_2, {\sigma_1^2\over n}+{\sigma_2^2\over m})$
$\Rightarrow \dfrac{\bar{X}-\bar{Y}-(\mu_1-\mu_2)}{\sqrt{{\sigma_1^2\over n}+{\sigma_2^2\over m}}}\sim N(0,1^2)$
$\Rightarrow P\left\{\bar{X}-\bar{Y}-z_{\alpha/2}\sqrt{{\sigma_1^2\over n}+{\sigma_2^2\over m}} < \mu_1-\mu_2 < \bar{X}-\bar{Y}+z_{\alpha/2}\sqrt{{\sigma_1^2\over n}+{\sigma_2^2\over m}}\right\}=1-\alpha$

$\sigma_1, \sigma_2$를 모르는 경우?
Recall that 

# Approximate Confidence Interval for the Mean of a Bernoullie random variable

# Evaluationg a Point Estimator
