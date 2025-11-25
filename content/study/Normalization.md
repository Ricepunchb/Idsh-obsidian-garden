---
title: Normalization
alias: 정규화
publish: true
tags:
  - Study
date: 2024-07-05
---


입력으로 들어갈 데이터의 분포를 일정하게 맞춰주는 과정
안정적이고 빠른 학습을 위해 사용

# Data Normalization (Data Scaling)
입력에 들어갈 데이터를 normalizing 하는 과정
모델 훈련은 아니고 데이터 전처리 과정
- min-Max normalization : 데이터를 0~1로 정규화
  $$x_{\text{scaled}}={x-\min \over \max-\min}$$
- Standardization : 데이터를 평균 0, 분산 1의 정규화
$$x_{\text{scaled}}={x-\mu\over\sigma}$$
# In-Layer Normalization
신경망 내부에서 다음 계층으로 넘어갈때의 입력을 정규화

# Batch Normalization
미니 배치 내에서 각 변수가 같은 분포를 갖도록 정규화
텍스트 분석 모델 RNN에서 사용

# Weight Normalization
데이터 대신 모델 weight를 정규화