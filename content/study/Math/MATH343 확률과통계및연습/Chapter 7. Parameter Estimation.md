---
title: Chapter 7

publish: true


date: 2024-07-05
tags:
  - Math
  - Statistics
---


# Maximum Likelihood Estimator

**Estimator** : unknown parameter $\theta$를 *estimate* 하는데 사용되는 statistic. 계획, 공식과 비슷한 개념이다.

관측된 데이터 $x_1,...,x_n$이 파라미터 $\theta$를 가진 분포에서 나왔다고 할 때, **maximum likelihood estimation**은 "이 데이터가 나올 확률(likelihood)을 가장 높게 만드는 $\theta$"를 고르는 방법이다.

>[!attention] Likelihood function
>$$L(\theta)=\prod^n_{i=1}p(x_i;\theta) \quad\text{(discrete)},\qquad L(\theta)=\prod^n_{i=1}f(x_i;\theta)\quad\text{(continuous)}$$
>$$\hat\theta_{MLE}=\arg\max_\theta L(\theta)$$

곱셈은 미분하기 번거로우니, $\log$를 취해 곱을 합으로 바꾼 **log-likelihood** $\ell(\theta)=\log L(\theta)$를 대신 최대화한다. $\log$는 단조증가함수라 $\arg\max$는 그대로 유지된다.

**풀이 순서**: $\ell(\theta)$를 $\theta$로 미분 → $0$으로 놓고 $\hat\theta$에 대해 풀기 → 이계도함수(또는 경계값)를 확인해서 실제로 최댓값인지 검증.

>[!example] 예) Bernoulli$(p)$에서 온 $x_1,...,x_n$
>PMF $P(X=x;p)=p^x(1-p)^{1-x}$, 이 중 성공(1)이 $k$개라 하면
>$$\ell(p)=\sum_i\left[x_i\log p+(1-x_i)\log(1-p)\right]=k\log p+(n-k)\log(1-p)$$
>$$\ell'(p)=\dfrac{k}{p}-\dfrac{n-k}{1-p}=0 \;\Longrightarrow\; \hat{p}_{MLE}=\dfrac{k}{n}$$
>즉 MLE는 그냥 표본에서 성공한 비율이다.

>[!example] 예) Exponential$(\lambda)$에서 온 $x_1,...,x_n$
>PDF $f(x;\lambda)=\lambda e^{-\lambda x}$
>$$\ell(\lambda)=\sum_i\log(\lambda e^{-\lambda x_i}) = n\log\lambda-\lambda\sum_ix_i$$
>$$\ell'(\lambda)=\dfrac{n}{\lambda}-\sum_ix_i=0 \;\Longrightarrow\; \hat\lambda_{MLE}=\dfrac{n}{\sum_ix_i}=\dfrac{1}{\bar x}$$
>표본평균의 역수. $\text{Exp}(\lambda)$의 평균이 $1/\lambda$였다는 걸 생각하면 자연스러운 결과다.

>[!example] 예) Uniform$(0,\theta)$에서 온 $x_1,...,x_n$
>PDF $f(x;\theta)=1/\theta$ for $0\le x\le\theta$이므로 likelihood는
>$$L(\theta)=\begin{cases}1/\theta^n & \theta\ge\max_ix_i \\ 0 & \text{else}\end{cases}$$
>미분이 아니라 $\theta$가 작을수록 $1/\theta^n$이 커지므로, 조건($\theta\ge\max_ix_i$)을 만족하는 가장 작은 $\theta$, 즉 $\hat\theta_{MLE}=\max_ix_i$ 에서 최대가 된다. 이처럼 미분으로 풀리지 않는 경우도 있다.

>[!tip] 딥러닝과의 연결
>모델을 Cross-Entropy loss로 학습하는 것 $\left(-\dfrac1N\sum_i\log p(x_i)\right)$은 log-likelihood $\sum_i\log p(x_i)$를 최대화하는 것과 동치다. 즉 "관측된 데이터를 최대한 그럴듯하게 만드는 파라미터를 찾는다"는 MLE의 원리가 곧 신경망 학습의 원리이기도 하다.

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

참값 $\theta$와 estimator $\hat\theta$가 있을 때, $\hat\theta$는 데이터(random sample)의 함수이므로 그 자체로 random variable이다. 그래서 "이 estimator가 얼마나 좋은가"도 확률적으로 평가해야 한다.

>[!attention] Bias
>$$\text{bias}[\hat\theta]=E[\hat\theta]-\theta$$
>평균적으로(=여러 번 표본을 뽑아 반복하면) $\hat\theta$가 참값 $\theta$에서 얼마나 벗어나는가. $\text{bias}=0$이면 **unbiased estimator**라 부른다.

>[!attention] Variance
>$$V(\hat\theta)=E\left[(\hat\theta-E[\hat\theta])^2\right]$$
>표본이 바뀔 때마다 $\hat\theta$ 값이 얼마나 들쭉날쭉한가.

두 개념을 하나로 합친 것이 **mean squared error**이며, 아래처럼 bias와 variance로 깔끔하게 분해된다.

>[!attention] MSE = Bias² + Variance
>$$\text{MSE}(\hat\theta)=E[(\hat\theta-\theta)^2]=\text{Bias}(\hat\theta)^2+V(\hat\theta)$$

>[!note] proof.
>$E[\hat\theta]=\mu_{\hat\theta}$ 라 하면,
>$$\begin{align} \text{MSE}(\hat\theta)&=E[(\hat\theta-\theta)^2]=E[((\hat\theta-\mu_{\hat\theta})+(\mu_{\hat\theta}-\theta))^2] \\ &=E[(\hat\theta-\mu_{\hat\theta})^2]+2(\mu_{\hat\theta}-\theta)\underbrace{E[\hat\theta-\mu_{\hat\theta}]}_{=0}+(\mu_{\hat\theta}-\theta)^2 \\ &=V(\hat\theta)+\text{bias}(\hat\theta)^2 \end{align}$$

즉 estimator를 고를 때는 bias와 variance를 동시에 낮은 값으로 유지할 수 없는 경우가 많고(**bias-variance tradeoff**), 편향이 조금 있더라도 분산이 훨씬 작다면 MSE 기준으로는 더 나은 estimator일 수 있다.
