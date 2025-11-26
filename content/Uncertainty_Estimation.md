---
title: Uncertainty Estimation
alias: 불확실성 탐지
publish: true
date: 2025-11-26
tags:
  - AI
  - LLM
---
# Uncertainty Estimation (불확실성 탐지) 
## 1. 개요
[[AI Fallback Strategy|Fallback]]을 언제 수행할지 결정하기 위해서는 **모델이 "자신의 답변에 얼마나 확신이 없는지"**
를 수치화해야 한다. 이를 불확실성 추정이라 한다. 
## 2. 핵심 지표 및 수식 
### 2.1. Maximum Softmax Probability (MSP) 
가장 단순한 방법이다. 모델의 마지막 출력층(Softmax)에서 가장 높은 확률값을 신뢰도로 쓴다. 
$$ \text{Confidence}(x) = \max_k P(y=k | x) $$
- **한계**: 딥러닝 모델은 틀릴 때도 자신만만하게(Overconfident) 틀리는 경향이 있어 정확하지 않다. 
### 2.2. Predictive Entropy (예측 엔트로피) 
모든 클래스에 대한 확률 분포가 얼마나 퍼져있는지를 측정한다. 분포가 평평할수록(모든 답이 고만고만할수록) 불확실하다. 
$$ H(y|x) = - \sum_{k=1}^{K} P(y=k | x) \log P(y=k | x) $$
- **판단**: $H(y|x) > \tau$ (임계값) 이면 Fallback을 트리거한다. 
### 2.3. Perplexity (PPL) 
- 생성형 모델용 LLM이 문장을 생성할 때 얼마나 "당황"했는지를 나타낸다. 문장의 확률의 역수다. 
$$ \text{PPL}(X) = \exp \left( - \frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i}) \right) $$
- PPL이 너무 높으면 모델이 횡설수설하고 있을 가능성이 높다. 
## 3. 고급 기법: Self-Consistency 
CoT(Chain of Thought)를 여러 번 수행하고, 가장 많이 등장한(Majority Vote) 답을 채택하거나, 답들이 서로 너무 다르면 Fallback한다. 
$$ \text{Consistency} = \frac{\text{Count}(\text{Most Frequent Answer})}{\text{Total Samples}} $$
