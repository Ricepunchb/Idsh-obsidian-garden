---
title: SVM
alias: Support Vector Machine

publish: true

tags:
  - Study
  - ML

date: 2024-07-05
---


# 1. 최대 마진 분류기 (Maximal Margin Classifier)
$\text{Let }X_i\in\mathbb{R}^p,\quad X_i=(x_{i1},x_{i2},\dots,x_{ip})\text{ Then,}$
$$d=c+w_1x_{i1}+\cdots+w_px_{ip}\quad\text{when}\sum^p_{i=1}w_i^2=1$$
이게 점과 선형경계 사이의 최단거리

**마진** : 판단 경계로부터 가장 가까운 점까지의 거리 or 모든 데이터 중 경계까지의 최소 거리

SVM은 몇몇 경계와 가까운 샘플만 고려함. 이게 다른 방법들과 SVM의 차별점

**지지샘플** (Support) : 모델 결정 샘플. SVM에서는 경계에 가까운 샘플. 보통 샘플들은 벡터이므로 지지벡터라고도 함

*SVM은 마진이 최대가 되는 경계를 선택하고자 함!*

# 2. 소프트 마진 Soft Margin
현실적으로 데이터에서 완벽한 선형 경계를 찾는것은 불가능하다
몇몇 샘플들이 선형경계를 넘는것을 허용함으로써 모델 안정성을 높임
$$\max_{\mathbf{w,\xi}} M \text{ subject to }y_i(c+w_1x_{i1}+\cdots+w_px_{ip})\ge M(1-\xi_i) \\ \text{ when }\sum^p_{i=1}w_i^2=1,\,\sum^n_{i=1}\xi_i\le V,\, \xi_i\ge0$$
$V$는 튜닝 파라미터로 전체 위반 정도를 조절
$\mathbf{w},c,\xi$는 모델 파라미터, 최적화로 결정됨

$$\Leftrightarrow \min_{\mathbf{w},c,\xi}\|\mathbf{w}\|^2 \text{ s.t. }y_i(\mathbf{w}^T\mathbf{x}_i-c)\ge1-\xi_i,\,\sum^n_{i=1}\xi_i\le V,\, \xi_i\ge0$$
$$\Leftrightarrow \min_{\mathbf{w},c}\left[\|\mathbf{w}\|^2+C\sum^n_{i=1}\max[0,1-y_i(\mathbf{w}^T\mathbf{x}_i-c)]\right]$$
$$\Leftrightarrow\min_{\mathbf{w},c}\left[\sum^n_{i=1}\max[0,1-y_i(\mathbf{w}^T\mathbf{x}_i-c)]+\lambda\|\mathbf{w}\|^2\right]$$
$\lambda,C,V$는 모델의 복잡성을 결정하는 튜닝 파라미터들
- $\lambda \uparrow \Rightarrow C \downarrow, V \uparrow$ 마진 넓음, 모델 간단함, Underfit 가능성
- $\lambda \downarrow \Rightarrow C \uparrow, V \downarrow$ 마진 좁음, 모델 복잡함, Overfit 가능성

# 3. Kernel
선형 경계를 적용할 수 없는 데이터들도 많다
선형 모델을 비선형 데이터에 적용하려면?
고차원의 변수를 생성해 모델에 포함한다

- **커널변환** : 커널을 통해 샘플 공간을 다른 차원으로 변환

두 샘플 사이의 내적에 대한 계산을 **커널 함수** $k(\mathbf{x}_i,\mathbf{x}_j)$ 로 바꾸어 샘플 공간을 더 높은 차원으로 변환

- 선형 커널 : $k(\mathbf{x}_i,\mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$ 선형 판단 경계
- 다항 커널 : $k(\mathbf{x}_i,\mathbf{x}_j) = (1+\mathbf{x}_i^T\mathbf{x}_j)^d$ 고차항을 포함, $d$는 튜닝 파라미터
- Radial basis function 커널 : $k(\mathbf{x}_i,\mathbf{x}_j)=\exp[{-\gamma\|x_i-x_j\|^2}]$ 무한한 차원 확장, $\gamma$는 튜닝 파라미터
