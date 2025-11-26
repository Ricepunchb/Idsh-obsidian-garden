---
title: Transformer
alias: 트랜스포머
publish: true
tags:
  - AI
  - Model
date: 2024-07-05
---
# Transformer

> “Attention is All You Need” (2017) 이후 8년이 지났지만, 
> MoE도, Retrieval도, Mamba도 결국 이 위에 얹는 부품일 뿐이다.

# 1. 전체 구조

```mermaid
%%{init: {'theme':'default'}}%% 
graph TD 
	Input[문장 입력<br/>"오늘 날씨 어때?"] 
	Emb[토큰 + 위치 정보 합치기]

	Input --> Emb 
	Emb --> Encoder[Encoder<br/>6~48층 반복] 
	Emb --> Decoder[Decoder<br/>6~48층 반복]

	Encoder --> EncOut[문장의 의미 벡터] 
	EncOut --> Decoder

	Decoder --> LMHead[다음 단어 예측 헤드] 
	LMHead --> Output[출력<br/>"맑아요!"]

	style Input fill:#FFF8F0,stroke:#D97706 
	style Emb fill:#E3F2FD,stroke:#1976D2 
	style Encoder fill:#FFE5D4,stroke:#E67E22 
	style Decoder fill:#E8F5E9,stroke:#388E3C 
	style LMHead fill:#FFE0B2,stroke:#F57C00 
	style Output fill:#FFF8F0,stroke:#D97706
```
# 2. Encoder Decoder 하나의 레이어 구조

```mermaid
%%{init: {'theme':'default'}}%%
graph TD 
	X[이전 층에서 온 문장 벡터]

	X --> LN1[LayerNorm<br/>정규화] 
	LN1 --> MHA[Self-Attention<br/>"서로 누가 중요한가?"] 
	MHA --> Add1[+ 잔차 연결]

	Add1 --> LN2[LayerNorm 다시] 
	LN2 --> FFN[Feed-Forward<br/>"생각 좀 더 깊게 해보자"] 
	FFN --> Add2[+ 잔차 연결]

	Add2 --> Out[다음 층으로 전달]

	style MHA fill:#FFE5D4,stroke:#E67E22 
	style FFN fill:#E8F5E9,stroke:#388E3C
```
**왜 이렇게 생겼을까?**
쉽게 비유하자면,
- Attention = “지금 이 단어는 문장 어디를 제일 신경 써야 해?”
- FFN = “이 단어 자체를 좀 더 깊이 생각해보자”
- 잔차 연결 = “기존 정보 잊지 말고 계속 기억해!”
- LayerNorm = “값들이 너무 커지거나 작아지지 않게 잡아줘”
# 3. 핵심 수식
$$
\begin{aligned} 
& \text{1. Attention — “누가 제일 중요해?”} \\[1em] 
& \quad \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V \\[.8em] 
& \quad → Q(질문), K(열쇠), V(값)라고 생각하면 쉬움 \\[.5em] 
& \quad → \frac{1}{\sqrt{d_k}}\text{ 는 “너무 큰 값 나오면 안 돼” 라는 보험} \\[2.5em] 
& \text{2. Multi-Head — “여러 관점에서 보자”} \\[1em] 
& \quad \text{8 - 32개의 머리가 동시에 다른 곳을 봄} \\ 
& \quad → \text{하나의 머리로는 놓칠 수 있는 관계도 다 잡음} \\[2.5em] 
& \text{3. FFN — “조용히 생각 좀 해보자”} \\[1em] 
& \quad \text{옛날:} \text{ReLU}(xW_1)W_2 \\ 
& \quad \text{지금(2025):} \text{SwiGLU} \text{ ← 성능 더 좋고 학습도 안정적} \\[2.5em] & \text{4. 위치 정보 — “순서가 중요하잖아”} \\[1em] 
& \quad \text{옛날(2017):} \text{sin, cos로 고정된 파도 모양} \\ 
& \quad \text{지금(2025):} \text{RoPE ← 회전시키는 방식} \\ 
& \quad → \text{32k, 100k 길이도 문제없음! (Llama, Qwen, Gemma 다 씀)} \\[2em] 
& \text{5. Pre-Norm — “이게 진짜 정답”} \\[1em] 
& \quad \text{LayerNorm을 Attention/FFN 앞에 둠} \\ 
& \quad → \text{100층 이상 쌓아도 학습이 안 터짐} 
\end{aligned}
$$

# 4. 실무에서 쓰이는 Transformer 변종
| 변종             | 핵심 변화점                            | 대표 모델                       |
| -------------- | --------------------------------- | --------------------------- |
| Pre-Norm       | LayerNorm을 sublayers 앞에           | GPT-3, Llama, PaLM, Gemma   |
| RMSNorm        | LayerNorm → 제곱합으로 정규화             | Llama, Mistral, Qwen2       |
| SwiGLU         | ReLU → SwiGLU 활성화                 | Llama-2, PaLM-2, Mixtral    |
| RoPE           | Sinusoidal → Rotary Embedding     | Llama, PaLM, Mistral, Gemma |
| GQA / MQA      | KV 헤드 공유 → 속도 2~4배                | Llama-2 70B, Mistral-7B     |
| Sliding Window | Full attention → window attention | Mistral, Phi-3              |
# 5. 한 줄 요약
>입력 → Embedding + RoPE → N번 반복 (Pre-Norm → Multi-Head Self-Attention → Residual → Pre-Norm → SwiGLU FFN → Residual) → LM Head 이 구조만 알면 MoE도, Retrieval도, Long-Context도 다 설명 가능

