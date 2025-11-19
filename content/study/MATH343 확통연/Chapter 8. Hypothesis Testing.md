---
title: Chapter 8

publish: true

tags:
  - Study
  - Statistics

date: 2024-07-05
---


가설검정

A statistical hypothesis is usually a statement about a set of parameteres of a population distribution.

Accept 도 좋지만 Do not reject라는 개념을 쓴다 (틀리진 않는데 맞다고 하기엔 증거불충분)

# Significance levels
우리가 모르는 $\theta$에 대해 $H_0$라는 가설을 세우자
$H_0 :=$ *null hypothesis*
"*Simple Hypothesis*" := $H_0$가 참일때 distribution을 완전히 설명하는 것
"*Composite Hypothesis*" := Simple hypothesis가 아닌것

"*Critical Region*" 
$\begin{align} &(X_1,X_2,...,X_n)\notin C & \text{ "Do not reject"} \\ &(X_1,X_2,...,X_n)\in C & \text{ "Reject" } \end{align}$

"*Type 1 Error*" := $H_0$가 사실 맞는데 Reject하는 경우
"*Type 2 Error*" := $H_0$가 사실 틀린데 Do not reject 하는 경우

# Test Concerning the Mean of a Normal Population
## Case of Known Variance
분산 $\sigma^2$만 알 때
Suppose $X_1,X_2,...,X_n$ sample that $X_i\sim N(\mu,\sigma^2)$
