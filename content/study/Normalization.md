---
title: Normalization
alias: 정규화
publish: true
date: 2024-07-05
tags:
  - Transformer
  - Optimization
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

# RMSNorm
[[Layer Normalization]](LN)에서 평균을 빼는 연산(re-centering)을 없애고, 크기(scale)만 맞추는(re-scaling) 방식. Llama, Mistral, Gemma 등 최신 LLM의 표준 정규화 기법이다.

>[!attention] 정의
> 은닉 상태 $\mathbf X\in\mathbb R^D$의 각 원소를 RMS(Root Mean Square)로 나눠 단위 RMS를 갖게 만든 뒤, 학습 가능한 스케일 $\gamma$를 곱한다.
> $$ \bar{\mathbf X}=\frac{\mathbf X}{\text{RMS}(\mathbf X)+\epsilon}\odot\gamma,\qquad \text{RMS}(\mathbf X)=\sqrt{\frac 1D\sum_{i=1}^D x_i^2} $$

- **LN과의 차이**: LN은 평균 $\mu$를 빼고 표준편차 $\sigma$로 나누는 두 통계량을 계산하지만($\hat x=(x-\mu)/\sigma$), RMSNorm은 평균을 빼지 않고 RMS 하나만 계산한다.
  - 평균을 빼는 연산이 실제 성능에 큰 영향을 주지 않는다는 경험적 관찰에 기반해 생략 → 통계량 계산이 절반으로 줄어 학습/추론이 더 빠르다.
  - LN의 $\beta$(shift)도 없다 — 스케일 $\gamma$만 남긴다.
- **왜 정규화만으론 부족한가**: 정규화 자체는 모든 차원을 똑같은 기준(단위 RMS)으로 맞춰버리기 때문에, 차원마다 원래 갖고 있던 크기 정보(학습된 scale)가 사라진다. 하지만 모든 feature가 똑같은 크기를 가져야 하는 건 아니다 — 어떤 차원은 크게, 어떤 차원은 작게 유지되는 게 더 유리할 수 있다.
- **$\gamma\in\mathbb R^D$의 역할**: 정규화가 지운 차원별 크기 정보를 학습을 통해 되돌려준다.
  - $\gamma_i>1$ → $i$번째 차원 증폭
  - $\gamma_i<1$ → $i$번째 차원 억제
  - $\gamma_i\approx 0$ → 해당 차원을 사실상 꺼버림
  - 즉 $\gamma$는 "정규화로 인한 안정성"과 "차원별로 다른 크기를 가질 자유"를 동시에 준다.
- **QK-Norm**: 최신 모델은 Attention에서 $Q$, $K$ 벡터를 내적하기 전에도 RMSNorm을 한 번 더 적용해, 내적 값의 크기를 제어하고 학습을 안정화한다.

> [!tip] Transformer에서의 위치
> Modern Transformer LM에서는 매 레이어의 Attention 직전, FFN 직전, 그리고 마지막 unembedding 직전까지 총 $2L+1$번 RMSNorm이 들어간다. 전체 아키텍처 흐름은 [[Transformer]] 참고.