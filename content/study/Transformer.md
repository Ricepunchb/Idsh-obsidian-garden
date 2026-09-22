---
title: Transformer
alias: 트랜스포머
publish: true
date: 2024-07-05
tags:
  - Transformer
  - Architecture
---
# 0. 개요
2017년 구글의 "Attention Is All You Need" 논문에서 제안된 아키텍처이다. 등장 이후 8년이 지났지만, 최신 기술인 MoE(Mixture of Experts), Retrieval(RAG), Mamba 등도 결국 이 구조 위에 얹어지는 부품일 뿐, 거대한 뼈대는 변하지 않았다. 

> [!abstract] 핵심 철학 
> **"복잡한 재귀(RNN)나 합성곱(CNN) 없이, 오직 어텐션(Attention)만으로 충분하다."**

# 1. 전체 구조
- Transformer는 크게 입력을 이해하는 **Encoder**와 출력을 생성하는 **Decoder**로 나뉜다. 
	- 최신 LLM인 GPT 계열은 Decoder만 사용하기도 한다.

```mermaid
%%{init: {'theme':'default'}}%%
graph TD
    Input["문장 입력<br/>(예: '오늘 날씨 어때?')"] 
    Emb["Embedding + Positional Encoding"]

    Input --> Emb
    Emb --> Encoder["Encoder Stack<br/>(6~48층 반복)"]
    Emb --> Decoder["Decoder Stack<br/>(6~48층 반복)"]

    Encoder --> EncOut["Context Vector<br/>(문장의 의미)"]
    EncOut --> Decoder

    Decoder --> LMHead["LM Head<br/>(Next Token Prediction)"]
    LMHead --> Output["최종 출력<br/>(예: '맑아요!')"]

    style Input fill:#FFF8F0,stroke:#D97706
    style Emb fill:#E3F2FD,stroke:#1976D2
    style Encoder fill:#FFE5D4,stroke:#E67E22
    style Decoder fill:#E8F5E9,stroke:#388E3C
    style LMHead fill:#FFE0B2,stroke:#F57C00
    style Output fill:#FFF8F0,stroke:#D97706
```
### 구조적 특징 
1. **입력 임베딩**: 텍스트를 벡터로 변환하고 위치 정보(Positional Encoding)를 더한다. 
2. **Encoder**: 입력 문맥을 깊이 있게 이해하여 '의미 벡터'를 생성한다. 
3. **Decoder**: Encoder가 만든 의미 벡터와 이전에 생성한 단어들을 참고하여 다음 단어를 예측한다.
# 2. 레이어 상세 구조 (Encoder/Decoder Block)

```mermaid
%%{init: {'theme':'default'}}%% 
graph TD 
	X["이전 층 입력 X"] 
	X --> LN1["LayerNorm 1<br/>(Pre-Norm)"] 
	LN1 --> MHA["Multi-Head Attention<br/>(상호 관계 파악)"] 
	MHA --> Add1["Add (Residual)<br/>X + Attention"] 
	Add1 --> LN2["LayerNorm 2<br/>(Pre-Norm)"] 
	LN2 --> FFN["Feed-Forward<br/>(SwiGLU 등)"] 
	FFN --> Add2["Add (Residual)<br/>X' + FFN"] 
	Add2 --> Out["다음 층으로 전달"] 
	
	style MHA fill:#FFE5D4,stroke:#E67E22 
	style FFN fill:#E8F5E9,stroke:#388E3C 
	style LN1 fill:#F5F5F5,stroke:#9E9E9E 
	style LN2 fill:#F5F5F5,stroke:#9E9E9E
```
- **LayerNorm**: 데이터 값들이 너무 커지거나 작아지지 않게 정규화하여 학습 안정을 돕는다. (Pre-Norm 방식이 대세) 
- **Self-Attention**: "현재 단어가 문장 내의 다른 어떤 단어와 연관되는가?"를 계산한다. (문맥 파악) 
- **Residual Connection (잔차 연결)**: $x + f(x)$ 형태로, 연산 전의 정보를 더해주어 정보 손실을 막고 학습을 돕는다. 
- **Feed-Forward Network (FFN)**: 어텐션으로 모은 정보를 바탕으로 심층적인 특징을 추출하고 처리한다. ("생각을 정리하는 단계")
# 3. 핵심 수식
[[Attention]] 메커니즘 사용
$$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V $$ - **$Q$ (Query)**: 질문 ("나랑 관련된 애 누구야?") 
- **$K$ (Key)**: 색인 ("나 여기 있어") 
- **$V$ (Value)**: 내용 ("내 정보는 이거야") 
- **$\sqrt{d_k}$**: 차원이 커져도 내적 값이 폭발하지 않도록 나누어주는 스케일링 팩터(Insurance).

### Feed-Forward Network (FFN) 
과거에는 ReLU를 썼으나, 최근엔 학습 안정성이 높은 **SwiGLU**를 주로 쓴다.
$$ \begin{aligned} \text{Old (ReLU)} &: \text{ReLU}(xW_1 + b_1)W_2 \\ \text{New (SwiGLU)} &: \text{Swish}(xW_G) \otimes (xW_1) W_2 \end{aligned} $$
- **게이트 구조의 의미**: up projection $U=xW_1$과 gate projection $G=xW_G$이 각자 독립적인 표현을 만들고, $\text{Swish}(G)=G\odot\sigma(G)$가 "얼마나 통과시킬지"를 스스로 결정하는 게이트 역할을 한다. 즉 $U$는 내용, $G$는 그 내용에 대한 자기 확신(self-gating)이다.
- **파라미터 수**: up/gate/down 세 개의 projection 행렬이 각각 $D\times F$ 크기이므로 레이어당 $3DF$개다. Gate가 추가된 만큼, 기존 2-layer FFN($F=4D$일 때 $8D^2$)과 파라미터 수를 맞추려고 보통 $F\approx 8D/3$로 축소해서 쓴다 ($3D\cdot\frac{8D}{3}=8D^2$).
### Positional Encoding (위치 정보) 
순서 정보가 없는 Attention에 위치 정보를 주입한다. 
- **Original (2017)**: $\sin, \cos$ 함수를 이용한 고정된 절대 위치 값. 
- **Modern (RoPE)**: **Rotary Positional Embedding**. 벡터를 회전시켜 상대적인 위치 관계를 보존한다. Llama, Qwen 등이 채택하여 수십만 토큰 길이도 처리가 가능하다.

#### RoPE가 "상대 위치"를 보존하는 이유 (유도)
RoPE의 목표는, 위치 $m$의 쿼리 $\mathbf q$와 위치 $n$의 키 $\mathbf k$의 내적이 **오직 상대 위치 $n-m$에만 의존**하게 만드는 함수 $f$를 찾는 것이다.
$$ \langle f(\mathbf q,m),f(\mathbf k, n)\rangle=g(\mathbf q,\mathbf k,n-m) $$
회전 행렬 $R_{m\theta}$로 벡터를 회전시키는 $f(\mathbf x,m)=R_{m\theta}\mathbf x$가 이 조건을 만족한다.
$$ \begin{align*}
\langle R_{m\theta}\mathbf q,R_{n\theta}\mathbf k\rangle&=(R_{m\theta}\mathbf q)^\top R_{n\theta}\mathbf k\\
&=\mathbf q^\top R_{m\theta}^\top R_{n\theta}\mathbf k\\
&=\mathbf q^\top R_{-m\theta}R_{n\theta}\mathbf k &\text{(}R_\alpha^\top=R_{-\alpha}\text{, 회전행렬의 전치=역회전)}\\
&=\mathbf q^\top R_{(n-m)\theta}\mathbf k &\text{(}R_\alpha R_\beta=R_{\alpha+\beta}\text{, 회전의 합성)}
\end{align*} $$
결과가 $n-m$에만 의존하므로, **절대 위치를 따로 학습하지 않아도 상대 위치 관계가 내적에 자연히 새겨진다.**

- 실제로는 head 차원 $H$를 2개씩 짝지어($H/2$쌍) 쌍마다 다른 각도 $\theta_i$로 회전시킨다.
  $$ \theta_i=\Theta^{-2i/H} $$
  - $\Theta$(보통 10,000, HF의 `rotary_base`)가 유일한 하이퍼파라미터로, 값이 클수록 저주파(느리게 도는) 쌍의 회전 주기가 길어져 더 먼 거리까지 구분할 수 있다.
  - $i$가 작을수록(고주파) 인접 위치 간 차이가 커 **근접 토큰 구분**에, $i$가 클수록(저주파) 천천히 회전해 **장거리 정보**를 인코딩한다.
- 값(Value) 벡터는 회전시키지 않는다 — 위치 정보는 "누구에게 집중할지"에만 영향을 줘야 하고, 실제로 전달되는 내용($V$)은 바뀔 필요가 없기 때문이다.
- 복소수로 보면, 짝 $[x_{2i},x_{2i+1}]$을 복소수 $z=x_{2i}+ix_{2i+1}$로 두었을 때 각도 $\theta$만큼 회전은 $z\cdot e^{i\theta}$ (오일러 공식 $e^{i\theta}=\cos\theta+i\sin\theta$)와 동일하며, 실제 구현에서는 $H\times H$ 행렬을 직접 만들지 않고 원소별 곱셈($\odot$)으로 처리한다.
# 4. 실무에서 쓰이는 Transformer 변종

| 컴포넌트 | 변화점 (Key Change) | 적용 모델 예시 |
| :--- | :--- | :--- |
| **Normalization** | **Pre-Norm / RMSNorm**<br>입력 값을 평균 0, 분산 1로 맞추되, 평균 연산을 빼서 속도를 높임 | Llama 2/3, Mistral, Gemma |
| **Activation** | **SwiGLU**<br>ReLU보다 미분 가능 구간이 부드러워 학습이 잘 됨 | Llama, PaLM, Mixtral |
| **Position** | **RoPE (Rotary)**<br>절대 위치 대신 회전 변환으로 상대 위치를 학습 | Llama, PaLM, Gemma |
| **Attention** | **GQA / MQA**<br>Key, Value 헤드 수를 줄여 메모리 절약 및 속도 2~4배 향상 | Llama-2 70B, Mistral |
| **Context** | **Sliding Window**<br>전체를 다 보지 않고 특정 윈도우만 보며 연산량 감소 | Mistral, Phi-3 |

> [!tip] 더 자세한 내용
> - RMSNorm의 정확한 공식과 LayerNorm과의 차이는 [[Normalization]] 참고
> - 레이어별 파라미터 수 / FLOPs / 추론·학습 메모리 계산은 [[Transformer Accounting]] 참고

# 5. 한 줄 요약
> **Modern Transformer Recipe:**
> 입력 $\to$ Emb(+RoPE) $\to$ $N$번 반복 [ RMSNorm $\to$ Attention(GQA) $\to$ Residual $\to$ RMSNorm $\to$ SwiGLU $\to$ Residual ] $\to$ LM Head