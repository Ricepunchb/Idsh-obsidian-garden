---
title: Knowledge Distillation
publish: true
date: 2025-11-25
tags:
  - AI
  - LLM
---
# Knowledge Distillation 정리

> **“큰 모델이 아는 걸 작은 모델에게 쏙쏙 넣어주는 마법”**  
> → 온디바이스 LLM, 엣지 AI의 핵심 기술

## 1. 왜 쓰는가? (Motivation)
- 70B급 모델 → 추론 속도 느리고 메모리 터짐 (모바일/엣지 불가능)
- 1~7B 작은 모델을 처음부터 학습 → 성능 너무 구림
- 해결책: **Teacher (큰 모델)의 지식을 Student (작은 모델)에게 압축 전송**

실무 사례 (2024~2025)
- Llama-3-70B → Llama-3-8B
- Mistral-Large → Mistral-7B / Nemotron-4-340B → MiniChat-24B 등

## 2. 핵심 아이디어: Dark Knowledge
Teacher는 정답뿐만 아니라 **오답들 사이의 미묘한 관계**까지 알고 있음  
→ Hinton이 “Dark Knowledge”라고 명명 (2015)

### 예시: “이 사진은 뭘까?”
| 클래스   | Hard Label (일반 학습) | Soft Label (Teacher, T=5) | 의미 |
|----------|--------------------------|-----------------------------|------|
| 강아지   | 1.0                      | 0.85                        | 정답 |
| 고양이   | 0.0                      | 0.12                        | 좀 비슷함 |
| 판다     | 0.0                      | 0.02                        | 약간 비슷 |
| 비행기   | 0.0                      | 0.0001                      | 전혀 아님 |

Student는 이 **미묘한 차이**까지 배우기 때문에 일반 학습보다 훨씬 일반화 잘 됨!


## 3.1. Knowledge Distillation 도식

```mermaid
%%{init: {'theme':'default'}}%%
graph TD
    A[입력 x]
    T[Teacher Model\n큰 모델 70B+]
    S[Student Model\n작은 모델 7B]

    LT[z^T Teacher logits]
    LS[z^S Student logits]
    PT[Soft Label p^T\nSoftmax T>1]
    PS[Soft Label p^S\nSoftmax T>1]
    GT[Ground Truth y\nOne-hot]

    KD[KD Loss\nT&#178; &times; KL(p^T &parallel; p^S)]
    CE[CE Loss\nCrossEntropy T=1]
    Total[Total Loss\n&alpha; &times; KD + (1-&alpha;) &times; CE]

    A --> T --> LT
    A --> S --> LS

    LT --> PT
    LS --> PS

    PT --> KD
    PS --> KD
    LS --> CE
    GT --> CE

    KD --> Total
    CE --> Total

    style A fill:#FFF8F0,stroke:#333,stroke-width:2px
    style T fill:#FFE5D4,stroke:#E67E22,stroke-width:3px
    style S fill:#E8F5E9,stroke:#388E3C,stroke-width:3px
    style LT fill:#FFF9C4,stroke:#F57C00
    style LS fill:#FFF9C4,stroke:#F57C00
    style PT fill:#E3F2FD,stroke:#1976D2
    style PS fill:#E3F2FD,stroke:#1976D2
    style GT fill:#FCE4EC,stroke:#AD1457
    style KD fill:#FFE0B2,stroke:#F4511E
    style CE fill:#E8F5E9,stroke:#43A047
    style Total fill:#E8EAF6,stroke:#3949AB,stroke-width:4px,font-weight:bold
```
## 3.2. 핵심 수식

$$
\begin{align} 
&\text{1. Temperature Softmax} \\ 
&\quad q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \\ 
&\quad T > 1 \rightarrow \text{분포가 부드러워짐} \quad \text{(실무: } T=3\sim20\text{)} \\[1em]

&\text{2. Distillation Loss (KL Divergence)} \\ 
&\quad \mathcal{L}_{KD} = T^2 \cdot \text{KL}(p^T \| p^S) 
= T^2 \sum_i p_i^T \log \frac{p_i^T}{p_i^S} \\ 
&\quad T² \text{ 곱하는 이유: T 커질수록 gradient magnitude가 1/T²로 작아지니까 보상} \\[1em]

&\text{3. Hard Label Loss (일반 CE)} \\ 
&\quad \mathcal{L}_{CE} = -\sum_i y_i \log p_i^S(T=1) \\[1em]

&\text{4. 최종 Loss} \\ 
&\quad \mathcal{L} = \alpha \cdot \mathcal{L}_{KD} + (1-\alpha) \cdot \mathcal{L}_{CE} \\ &\quad \alpha = 0.7\sim0.9 \text{ 많이 씀} 
\end{align} 
$$
## 4. KD 종류 한눈에 비교

| 종류                | 무엇을 따라하냐?                    | 장점                  | 대표 논문/실무 예시               |
| ----------------- | ---------------------------- | ------------------- | ------------------------- |
| Response-based    | 최종 softmax 출력                | 가장 간단, 효과 좋음        | 원조 Hinton (2015), 대부분 실무  |
| Feature-based     | 중간 레이어 feature map           | 더 많은 정보 전달          | FitNet, AT, RKD           |
| Relation-based    | 샘플 간 관계 (Gram matrix 등)      | 구조적 지식 전달           | CRD, VID                  |
| Logits-based      | Temperature 없이 raw logits 맞춤 | T 없이도 잘 됨           | MiniLLM, TextKD           |
| Self-Distillation | Teacher = Student (같은 모델)    | 추가 Teacher 없이도 성능 ↑ | Llama-3-8B 자체 KD, Gemma-2 |

## 5. 2025년 실무 팁 & 트렌드

- MiniLM, DistilBERT 스타일 → 이미 기본
- LLM 시대 → **Logits Matching + Continued Pre-training** 조합이 대세
- Teacher 없이도 KD 가능 → **Self-Distillation + Synthetic Data** (Gemma-2, Phi-3)
- 수학/코딩 특화 작은 모델 만들 때: **DeepSeek-Math-7B = 671B Teacher로 KD**
- 온디바이스 최적화: **Speculative Distillation** (작은 모델이 큰 모델 흉내 내면서 동시에 draft 토큰 생성)

## 한 줄 요약
> “큰 모델이 아는 모든 걸 온도 올려 부드럽게 만들어 작은 모델한테 KL로 쑤셔 넣는 기술”

