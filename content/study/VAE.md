---
title: VAE
alias: Variational Auto Encoder

publish: true

tags:
  - Study
  - AI

date: 2024-07-05
---


# 개요
- 잠재변수가 특정확률분포 (보통 $N(0,1^2)$) 을 갖도록 AE 구성
- 특정확률분포에서 샘플을 생성하고, 디코더에 넣어 새로운 데이터 생성
## Remark
AE에서는,
- 잠재 공간에서의 주어진 값을 디코더에 넣어 데이터를 생성
- 잠재변수의 Distribution에는 관여하지 않음
- 잠재공간에서 새로운 값을 추출하기 어려움

# 구조
1. 각 $x_i$는 Encoder를 통해 $N(\mu_i,\sigma_i^2)$로 mapping
2. $\mu=(\mu_1,\mu_2,...,\mu_n), \sigma=(\sigma_1,\sigma_2,...,\sigma_n)$
3. $N(\mu,\sigma^2)$으로 부터 random sample $z$ 를 임의 추출
4. Decoder를  통해 $z$ 를 $\hat{x}$으로 재구성
## Training
$$Loss = \|x-d(z)\|^2+KL\left(N(\mu,\sigma^2),N(0,1^2)\right)$$
$KL$은 Kullback-Leibler divergence로, 두 분포 사이의 distance
- 같은 $X$에 대해 매번 다른 $z$가 생성되어 $\hat{x}$를 생성
# 장단점
- 생성모델에 대한 수학적 접근과 확률분포 근사에 대한 접근법
- AE보다 잠재공간이 좀 더 밀집되어 있어 생성에 유리함
- 다른 생성모델에 비해 성능이 떨어짐
