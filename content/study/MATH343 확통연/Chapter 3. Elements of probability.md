---
title: Chapter 3

publish: true

tags:
  - Study
  - Statistics

date: 2024-07-05
---



$S$ := sample space, 나올 수 있는 모든 산출의 집합.
$E$ := Event, sample space의 부분집합. 관심있는 경우.

# 4. Axioms of probability

^ec8f65

  A1. $0 \leq P(E) \leq 1 \qquad \forall E \subset S$
  A2. $P(S)=1$  
  A3. $\displaystyle E_i\cap E_j=\emptyset \Rightarrow P(\bigcup^n_{i=1}E_i) = \sum^n_{i=1}P(E_i)$

이 세 Axiom을 다 만족하면 $P$는 확률 함수다. 

## Propostion 1.
$$P(E^c)=1-P(E)$$
### ___proof.___ 
Since $S=E\cup E^c,\quad E\cap E^c=\emptyset.$ 
By A3, $P(S)=P(E\cup E^c)=P(E)+P(E^c)$.
By A2, $P(S)=1 \Rightarrow 1=P(S)=P(E\cup E^c) \Rightarrow 1-P(E)=P(E^c)$

## Proposition 2.
$$ P(E\cup F)=P(E)+P(F)-P(E\cap F)$$
### ___proof.___
Suppose that $E\cap F=\emptyset$. Then by A3, $P(E\cup F)=P(E)+P(F)-P(E\cap F).$
Now, suppose that $E\cap F =A \ne \emptyset$. Let $E_1 = E/A$ and $F_1 =F/A.$ 
$\Rightarrow P(E\cup F)=P(E_1\cup F_1\cup A)=P(E_1\cup A)+P(E_1)=P(E_1)+P(F_1\cup A).$
$\begin{align} \Rightarrow 2P(E\cup F) &= P(E)+P(F)+P(F_1)+P(E_1) \\ &=P(E)+P(F)+(P(F)-P(A))+(P(E)-P(A)) \\ &= 2P(E)+2P(F)-2P(E\cap F). \end{align}$
$\therefore P(E\cup F)= P(E)+P(F)-P(E\cap F)$

# 6. Conditional probability
$$ P(E|F) = \dfrac{P(E\cap F)}{P(F)} $$
F가 일어났을때, E가 일어났을 확률.
F안에 E가 차지하는 비중.

# 7. Bayes' formula

Let E, F be events. Then from $P(E|F)=\dfrac{P(E\cap F)}{P(F)}$.
$P(E\cap F)=P(F)P(E|F)=P(E)P(F|E)$

$\Rightarrow P(E|F)=\dfrac{P(E)P(F|E)}{P(F)}$   ... _ver 1_

$\begin{align} P(E)&=P(F)P(E|F)+P(F^c)P(E|F^c) \\ &=P(F)P(E|F)+(1-P(F))P(E|F^c)\end{align}$    ..._ver 2_

**In general case,**
Suppose  that $F_1, F_2, \cdots F_n$ are mutually exclusive and $\displaystyle \bigcap^n_{i=1}F_i=S$.
Then, $\displaystyle E=\bigcup^n_{i=1}(E\cap F_i)$ $\displaystyle \Rightarrow P(E)=\sum^n_{i=1}P(E\cap F_i)=\sum^n_{i=1}P(E|F_i)P(F_i).$
$\displaystyle \Rightarrow P(F_j|E)=\frac{P(E\cap F_j)}{P(E)}=\dfrac{P(E|F_j)P(F_j)}{\sum^n_{i=1}P(E|F_i)P(F_i)}$


# 8. Independent Events
"E is independent of F" $\Leftrightarrow$ "$E \perp F$"
$\Leftrightarrow P(E|F)=P(E|F^c)$
$\Leftrightarrow P(E|F)=P(E)$
$\Leftrightarrow P(E\cap F)=P(E)P(F)$

## Proposition. 
$$ E \perp F \Rightarrow E \perp F^c $$
### ___proof.___
$P(E)=P(E\cap F)+P(E\cap F^c)$
$\begin{align} &\Rightarrow P(E\cap F^c)=P(E)-P(E\cap F) = P(E)-P(E)P(F) \\ &\Rightarrow P(E)(1-P(F))=P(E)P(F^c) \\ &\Rightarrow E \perp F^c \end{align}$

## Proposition.
$$ A\cap B = \emptyset \not\Rightarrow A\perp B$$
### ___proof.___
By definition of independent, $P(A\cap B) = P(A)P(B)$ holds.
By assumption, $P(A\cap B)=P(\emptyset)=0.$ So $P(A)=0 \bigvee P(B)=0$.
However, there are many cases that $P(A)\ne0 \bigwedge P(B)\ne0 \bigwedge P(A\cap B)=0$.