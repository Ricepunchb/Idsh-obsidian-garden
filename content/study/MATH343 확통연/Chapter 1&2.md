---
title: Chapter 1 & 2

publish: true

tags:
  - Study
  - Statistics

date: 2024-07-05
---



# Chapter 1 & 2
**population** : In statistics, we are interested in obtaining information about a total collection of elements, which we will refer to as the "*population*"

### __Def.__ "Sample mean" := $\bar{x} = \dfrac{x_1+x_2+\cdots + x_n}{n}$
#### __Def.__ "Sample variance" $:= s^2 = \displaystyle \sum_{i=1}^n \dfrac{(x_i-\bar{x})^2}{n-1}$

#### __Def.__ "Sample standard deviation" $:= s = \displaystyle \sqrt{\sum^n_{i=1}\frac{(x_i-\bar{x})^2}{n-1}}$

#### __Def. 2.4.__ "*Chebyshev's inequality*" 
$$ s^2 = {1\over n-1}\sum^n_{i=1}(x_i-\bar{x})^2 \quad \wedge s_k = \{i|i\in [1,n]\subset \mathbb{N}\ : |x_i-\bar{x}|<ks \} \quad (k\geq 1) $$
$$ \Rightarrow {|s_k| \over n} \geq 1-{n-1 \over nk^2} > 1-{1\over k^2} $$
#### ___proof.___ 
$\begin{align} \displaystyle (n-1)s^2 &= \sum^n_{i=1}(x_i-\bar{x})^2 = \sum_{i\in s_k}(x_i-\bar{x})^2 + \sum_{i \not \in s_k}(x_i-\bar{x})^2 \\ &\geq \sum_{i \not\in s_k}(x_i-\bar{x})^2 \end{align}$
 Since $i\not\in s_k \Leftrightarrow |x_i-\bar{x}|\geq ks$ ,
$$ \geq \sum_{i\not\in s_k}(ks)^2 = (ks)^2|s^c_k|=k^2s^2(n-|s_k|) $$
$\begin{align} &\Rightarrow (n-1) \geq k^2(n-|s_k|) \\ &\Rightarrow {n-1 \over k^2} \geq n-|s_k| \\ &\Rightarrow {|s_k| \over n} \geq 1-{n-1 \over nk^2} > 1-{1 \over k^2}  \qquad \blacksquare \end{align}$ 

## 2.6 Paired datasets and the sample correlation coefficient.
**Def.** The smaple correlation coefficient $r$
$$ r = \sum^n_{i=1}{(x_i-\bar{x})(y_i-\bar{y}) \over (n-1)s_xs_y} \qquad -1\leq r \leq 1$$
Correlation measures association, not causation.




