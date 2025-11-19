---
title: Chapter 4

publish: true

tags:
  - Study
  - Statistics

date: 2024-07-05
---



보통 Sample space를 $\Omega$ 로 나타냄.
Random variable X=1 이나 Y=3 이런것들은 변수를 말하는게 아니라 함수와 같은 역할을 함.
### Remark
$P$는 [[Chapter 3. Elements of probability#^ec8f65|Probability function]].


# 2. Types of random variable

## Cumulative distribution function

$$F(x)=P\{X\leq x\}$$
## Probability mass function

주사위나 동전 던지기 같은 것. Sample space가 유한이 아니라 countable이기만 하면 됨.

$p(a) = P\{X=a\}$ **Probability mass function** 이라고 함. 

$\displaystyle\sum^n_{i=1}p(x_i)=1$ 을 만족.

## Probability density function

키, 몸무게, 어떤 실수든지 변수가 될 수 있는것을 Sample space로 가짐.
e.g. $P(3<X<5)=\frac{1}{2}$ 그런데, $P(X=4.12) \approx 0$. 즉, 특정 점의 확률은 0이 됨.
마치 화살과녁에 특정 점에 화살을 맞추는건 불가능한 것처럼.

$X$ 가 continuous random variable이면, 
$$P\{X=a\} = \int^a_af(x)dx $$
인 실수 전역에서 nonnegative한 $f(x)$가 존재한다. 이 함수가 바로 **Probability density function** 이다.

e.g. $\displaystyle P(1<X<2)=P(X\in(1,2))=\int^2_1f(x)dx$.   $f$가 p.d.f. 임
성질들은 다음과 같음.
1. $\displaystyle P(X=1)=\int^1_1f(x)dx=0.$
2. $\displaystyle \int^{\infty}_{-\infty}f(x)dx=1,\quad f(x)\geq 0$
3. $\displaystyle P(a\leq X\leq b)=P(a<X\leq b)=P(a\leq X<b)=P(a<X<b)=\int^a_bf(x)dx.$

### Examples
Let X be continuous random variable and p.d.f $f(x)=\begin{cases} C(4x-2x^2), & 0<x<2 \\ 0, & \text{else} \end{cases}$

$\displaystyle 1=\int^{\infty}_{-\infty}f(x)dx=\int^2_0f(x)dx=\int^2_0C(4x-2x^2)dx$
$\displaystyle =C(2x^2-\frac{2x^3}{3})|^2_0 = \frac{8}{3}C$
$\Longrightarrow C=\dfrac{3}{8}$

Now find $P(X>1)$
$\displaystyle P(X>1)=\int^{\infty}_1f(x)dx=\int^2_1f(x)dx$


# 3. Jointly distributed random variables

## Joint cummulative probability distribution function
$$ F(x,y)=P(X\leq x, Y\leq y) $$
$$F_X(x)=P(X\leq x)=P(X\leq x, Y\in\mathbb{R})=F(x,\infty)$$
$$F_Y(y)=P(Y\leq y)=P(X\in\mathbb{R},Y\leq y)=F(\infty, y) $$

## Joint p.m.f. 
of X and Y

$X,Y$가 discrete random variables 이면,

$$p(x_i, y_j) = P\{X=x_i, Y=y_j\} $$
$$p_X(x_i)=P\{X=x_i\}=\sum_jP\{X=x_i, Y=y_j\} = P(\bigcup_j\{X=x_i, Y=y_j\})$$ similar in $p_Y(y_j)$

## Joint p.d.f 
of X and Y

 $X,Y$ are ___jointly continuous___ if there exists $f(x,y)$ for all real $x$ and $y$ such that
$$ P\{(X,Y)\in C\} = \iint_{(x,y)\in C}f(x,y)dxdy$$ note that $C=\{(x,y):x\in A, y\in B\}\in A\times B$

$$\Leftrightarrow\, P\{X\in A, Y\in B\} = \int_B\int_Af(x,y)dxdy$$

Recall  ___joint cumulative probability distribution function___ $F(a,b)$.
$$\begin{align} F(a,b)&=P\{X\leq a, Y\leq b\} = P\{(X,Y)\in (-\infty,a]\times (-\infty, b]\} \\ &=\int^b_{-\infty}\int^a_{-\infty}f(x,y)dxdy\end{align}$$ $$\Rightarrow f(x,y) = \frac{\partial^2}{\partial x\partial y}F(x,y) $$

Notation : $\displaystyle f_X(x)=\int^{\infty}_{-\infty}f(x,y)dy$

## Independent random variables
### Definition
Suppose that X and Y are *independent.* $$ X \perp Y \Longleftrightarrow P\{X\in A, Y\in B\} = P\{X\in A\} P\{Y\in B\}$$ Then,
$$ F(a,b) = P\{X\leq a, Y\leq b\}=P\{X\leq a\} P\{Y\leq b\} = F_X(a)F_Y(b)$$
$$ p(x,y) = P\{X=x, Y=y\} = P\{X=x\}P\{Y=y\}=p_X(x)p_Y(y)$$
$$ f(x,y) = f_X(x)f_Y(y)$$

## B. Conditional distributions

Recall that conditional probability $P(E|F) = \dfrac{P(E\cap F)}{P(F)}$.

If X and Y are discrete random variables,
### Conditional p.m.f.
  of $X$ for given $Y=y$
$$ p_{X|Y}(x|y) = P\{X=x|Y=y\} = {P\{X=x,Y=y\}\over P\{Y=y\}} = {p(x,y)\over p_Y(y)}$$

If $X$ and $Y$ are _jointly continuous_ and have _joint probability density function_ $f(x,y)$,
### Conditional p.d.f.
of $X$ for given $Y=y$
$$ f_{X|Y}(x|y)={f(x,y)\over f_Y(y)}$$

# 4. Expectation

Notation. $\text{expectaion, expected value of} X := E[X]$

If X is discrete random variable, $$ E[X]=\sum_ix_iP\{X=x_i\}=\sum_ix_ip(x_i) $$
If X is continuous random variable, $$ E[X] = \int^{\infty}_{-\infty}xf(x)dx $$

### 유도과정
for $X=x_i,\quad x_iP\{x_i\leq X \leq x_{i+1}\},\quad x_{i+1}=x_i+\Delta x$.
$\displaystyle \begin{align} \sum x_iP\{x_i\leq X\leq x_{i+1}\} \leq E[X] &\leq \sum x_{i+1}P\{x_i\leq X \leq x_{i+1}\} \\ &=\sum x_iP\{x_i\leq X \leq x_{i+1}\}+\sum \Delta xP\{x_i\leq X \leq x_{i+1}\} \end{align}$ 
$\displaystyle \begin{align} &\Longrightarrow \sum x_iP\{..\} \leq E[X] \leq \sum x_iP\{..\}+\Delta x \\ &\Longrightarrow \Delta x\rightarrow 0. \quad E[X]\rightarrow\sum x_iP\{..\} \\ &\Longrightarrow E[X]=\sum^{\infty}_{i=0}x_iP\{..\} = \sum^{\infty}_{i=0}\int^{x_{i+1}}_{x_i}x_i f(t)dt \rightarrow \int^{\infty}_{-\infty}xf(x)dx \end{align}$

# 5. Properties of the expected value

$$ E[X^2]=\sum x_i^2 P\{X=x_i\}$$

여기서 $X^2$ 은 합성함수 $X\circ X$가 아니라, $X(w)=x$일때 $X^2(w)=x^2$ 인 함수를 뜻한다.


### Proposition 1. 
**Expectation of a function of a random variable.**
- X가 discrete random variable 일때.
  $\displaystyle E[g(X)]=\sum_xg(x)p(x)$
- X가 continuous random variable 일때.
  $\displaystyle E[g(X)]=\int^{\infty}_{-\infty}g(x)f(x)dx$
  
### Corollary 2.
If $a, b$ constants, then 
$$\displaystyle E[aX+b]=aE[X]+b$$

___proof.___
If discrete, 
$\displaystyle\begin{align} E[aX+b]&=\sum_x(ax+b)p(x) = a\sum_xxp(x)+b\sum_xp(x) \\&=aE[X]+b \end{align}$
If continuous,
$\displaystyle\begin{align} E[aX+b]&=\int^{\infty}_{-\infty}(ax+b)f(x) = a\int^{\infty}_{-\infty}xf(x)+b\int^{\infty}_{-\infty}f(x) \\&=aE[X]+b \end{align}$

By this, $g(x)=x^n \quad\Longrightarrow\quad E[g(X)]=E[X^n]:\text{nth moment of }X$


## A. Expected Value of sums of random variables
$$
\begin{aligned}
E[g(X, Y)] & =\sum_y \sum_x g(x, y) p(x, y) \quad \text { in the discrete case } \\
& =\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} g(x, y) f(x, y) d x d y \quad \text { in the continuous case }
\end{aligned}
$$

For example, if $g(x,y)=x+y$,
$\displaystyle\begin{align} E[X+Y]&=\int^{\infty}_{-\infty}\int^{\infty}_{-\infty}(x+y)f(x,y)dxdy \\ &=\int^{\infty}_{-\infty}\int^{\infty}_{-\infty}xf(x,y)dxdy+\int^{\infty}_{-\infty}\int^{\infty}_{-\infty}yf(x,y)dxdy \\ &= E[X]+E[Y] \end{align}$

In general,
$E[X_1+X_2+\cdots+X_n]=E[X_1]+E[X_2]+\cdots+E[X_n]$


# 6. Variance

### Definition.
$X$ is random variable with mean $\mu$, then the _variance_ of $X$ is 
$$V(X)=E[(X-\mu)^2]=E[X^2]-\mu^2$$

For any constants $a, b$
$$V(aX+b)=a^2V(X)$$


# 7. Covariance and variance of sums of random variables

### Definition.
Covariance of X, Y. Notation : $Cov(X,Y)$
$$Cov(X,Y) = E[(X-\mu_x)(Y-\mu_y)]$$

성질들
1. $Cov(X,Y)=E[XY]-E[X]E[Y]$
2. $Cov(X,X)=V(X)$
3. ${Cov}(aX, Y)=a{Cov}(X,Y)$ for any constant $a$

### Lemma 1.
$Cov(X_1+X_2, Y)=Cov(X_1,Y)+Cov(X_2,Y)$

### Thm.
$$ X, Y \\ \text{are independent} \Longrightarrow \\ Cov(X,Y)=0 $$

일반적으로, $Var(X+Y) \ne \\ Var(X)+Var(Y)$ 이다. 성립은 $X,Y$가 independent 일때만 가능.

### Proposition 2.
$$ Cov(\sum_{i=1}X_i,\sum_{j=1}Y_j)=\sum_i\sum_jCov(X_i,Y_j) $$

### Corollary 3.
$$ V(\sum_{i=1}X_i)=\sum_{i=1}V(X_i)+\sum_{i=1}\sum_{j\ne i}Cov(X_i,X_j) $$
$$ Corr(X,Y)=\dfrac{Cov(X,Y)}{\sqrt{V(X)V(Y)}} $$


# 8. Moment generating function
$\displaystyle E[g(x)]=\int^{\infty}_{-\infty}g(x)f(x)dx, \quad g(x)=e^{tx}\Longrightarrow g(X)=e^{tX}$

### Definition
Moment generarting function of X := $\phi (t)$
$$\phi(t)=E[e^{tX}]=\begin{cases} \sum_i e^{tx_i}p(x_i) & X \text{ is discrete} \\ \int^{\infty}_{-\infty}e^{tx}f(x)dx & X \text{ is continuous} \end{cases} $$
$$ \phi'(t) = E[Xe^{tX}]$$
Hence $\phi'(0)=E[X]$
$$\phi''(t)=E[X^2e^{tX}] $$
Hence $\phi''(0)=E[X^2]$
$\phi^{(n)}(0)=E[X^n]$

Let $Z=X+Y$, then $\phi_Z(t)=E[e^{tX}e^{tY}]$
Hence, $X\perp Y \Leftrightarrow \phi_X(t)\phi_Y(t)$


# 9. Chebyshev's inequality and the weak law of large numbers

### Proposition 1. (Markov's inequality)
X is random variable that takes only nonnegative values $\Rightarrow \text{for any value } a>0, \\P\{X\ge a\}\le\dfrac{E[X]}{a}$

### Proposition 2. (Chebyshev's inequality)
If X is a random variable with mean $\mu$ and variance $\sigma^2$,
$\Rightarrow \text{for any value }k>0, \\ P\{|X-\mu|\ge k\}\le\dfrac{\sigma^2}{k^2}$

### Thm 3. (The weak law of large numbers)
Let $X_1,X_2,\cdots,$ be the sequence of independent an identically distributed random variables, each having mean $E[X_i]=\mu$.
Then, for any $\epsilon>0$,
$$P\left\{\left|\dfrac{X_1+X_2+\cdots+X_n}{n}-\mu\right|>\epsilon\right\}\\\rightarrow\\ 0 \text{ as }n\\\rightarrow\\\infty$$
