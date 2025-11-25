---
title: 생성형 모델
alias: Generative Model
publish: true
tags:
  - AI
date: 2024-07-05
---




# 개요
## Determinative Model
판별모델
### Supervised
- $x$가 주어지면 라벨 $y$에 관심 있음
- 데이터 형태 $(x,y)$
- $x, y$간의 관계 함수 도출이 주요 문제
- 예) Classification, Regression 등

### Unsupervised
- $X$의 구조를 파악
- Unlabeled, $y$가 없음
- $x$의 distribution이나 구조에 관심있음
- 예) Clustering, Dimension Reduction 등

## Generative Model
- $x$가 주어지면 그것과 비슷한 $x'$를 새로 생성
- $x$가 데이터
- $x$의 distribution을 파악하고 거기서 sample 생성
- 예) 이미지 생성, 문장 생성 
# A. [[VAE]]
Variational Autoencoder
변이형 오토인코더

# B. [[GAN]]
Generative Adversarial Network
적대적 생성모델
