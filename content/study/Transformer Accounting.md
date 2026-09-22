---
title: Transformer Accounting
alias: 트랜스포머 연산량/메모리 계산
publish: true
date: 2026-09-22
tags:
  - Transformer
  - Inference
---
# 1. 개요
"이 모델은 파라미터가 몇 개고, 학습/추론에 FLOPs와 메모리가 얼마나 드는가?"를 하이퍼파라미터만으로 손으로 계산하는 방법이다. [[Transformer]]의 구조(RMSNorm → GQA Attention → RoPE → SwiGLU FFN)를 그대로 따라가며 각 단계에 몇 개의 곱셈-덧셈이 들어가는지 세는 것뿐이라, 구조를 이해하고 나면 계산 자체는 기계적이다.

> [!abstract] 핵심 철학
> **모델 하나를 통째로 이해하려 하지 말고, "레이어 하나 안에서 어떤 shape의 행렬이 어떤 shape과 곱해지는가"만 추적하면 전체 계산량이 나온다.**

# 2. 표기법
| 기호 | 의미 |
| :--- | :--- |
| $B$ | 배치 내 시퀀스 수 |
| $L$ | 레이어 수 |
| $T$ | 생성할 토큰 수 (decode 길이) |
| $S$ | 주어진 문맥 길이 (prefill 길이) |
| $V$ | vocab size |
| $D$ | hidden dimension |
| $H$ | head dimension |
| $F$ | MLP hidden dimension (보통 $F\approx 8D/3$) |
| $N$ | Query 헤드 수 ($N\cdot H=D$) |
| $K$ | Key/Value 헤드 수 ([[Attention#7. GQA & MQA (헤드 수를 줄이는 변형)\|GQA]]에서 $K<N$) |
| $G$ | GQA 그룹 크기 $=N/K$ |

prefill 단계에서는 관례상 $S=T$로 두고 계산한다(주어진 문맥 전체를 한 번에 forward하는 상황).

# 3. 파라미터 수
레이어 하나당 파라미터 수를 구성 요소별로 센다.

| 구성 요소 | shape | 파라미터 수 |
| :--- | :--- | :--- |
| 토큰 임베딩 | $(V,D)$ | $VD$ |
| Attention $Q$ | $(D,D)$ | $D^2$ |
| Attention $K$ | $(D,KH)$ | $DKH$ |
| Attention $V$ | $(D,KH)$ | $DKH$ |
| Attention $O$ | $(D,D)$ | $D^2$ |
| FFN up | $(D,F)$ | $DF$ |
| FFN gate | $(D,F)$ | $DF$ |
| FFN down | $(F,D)$ | $DF$ |
| RMSNorm ×2 (attn 전, FFN 전) | $D$ each | $2D$ |
| Unembedding | $(D,V)$ | $VD$ |

- Attention 총합: $2D^2+2DKH\approx 4D^2$ (표준 MHA처럼 $K=N$이면 $KH=D$이므로 $2D^2+2D^2=4D^2$).
- FFN 총합: $3DF$.
- 레이어 하나: $4D^2+2D+3DF$.
- **전체 모델**:
$$ \text{Params} = 2VD + L(4D^2+2D+3DF) \approx 2VD+12LD^2 $$
  ($F=8D/3$일 때 $3DF=8D^2$이므로 $4D^2+8D^2=12D^2$, $2D$는 $D^2$에 비해 무시 가능할 만큼 작음.)

```python
num_params = sum(p.numel() for p in model.parameters())
```

# 4. 순전파 FLOPs (Forward Pass)
행렬곱 $(m,n)\times(n,p)$는 $2mnp$ FLOPs (곱셈+덧셈)라는 규칙 하나로 전부 유도된다. prefill 기준($S=T$)으로 레이어 하나당:

### 4.1. Attention
| 연산 | shape | FLOPs |
| :--- | :--- | :--- |
| $Q$ projection | $(B,S,D)\times(D,D)$ | $2BSD^2$ |
| $K$ projection | $(B,S,D)\times(D,KH)$ | $2BSDKH\approx 2BSD^2$ ($K=N$ 가정) |
| $V$ projection | $(B,S,D)\times(D,KH)$ | $2BSDKH\approx 2BSD^2$ |
| $O$ projection | $(B,S,D)\times(D,D)$ | $2BSD^2$ |
| $QK^\top$ | $(B,N,S,H)\times(B,N,H,S)$ | $2BNS^2H=2BS^2D$ ($D=NH$) |
| $AV$ | $(B,N,S,S)\times(B,N,S,H)$ | $2BS^2D$ |

합: $8BSD^2+4BS^2D$

### 4.2. FFN
up/gate/down 세 projection 모두 $(B,S,D)\times(D,F)$ 꼴이므로 $6BSDF$. $F=8D/3$이면 $6BSD\cdot\frac{8D}{3}=16BSD^2$.

### 4.3. 레이어 합계와 전체
$$ \text{레이어당} = 8BSD^2+4BS^2D+16BSD^2 = 2BSD(12D+2S) $$
$$ \text{Unembedding} = 2BSDV $$
$$ \text{전체 forward} = 2LBSD(12D+2S)+2BSDV \approx 2BSD(12LD+2LS+V) $$

> [!tip] 유명한 "$2ND$" 어림
> $S\ll D$인 짧은 문맥에서는 $2S$ 항이 무시되어 $\text{forward FLOPs}\approx 2BS\cdot(12LD^2)\approx 2\cdot(\text{토큰 수})\cdot(\text{파라미터 수})$가 된다. 학습 FLOPs를 어림할 때 흔히 쓰는 "$6ND$" 규칙(아래 5장)의 근거가 여기서 나온다.

# 5. 역전파 FLOPs
역전파는 forward의 **약 2배**로 어림한다.
- 파라미터에 대한 그래디언트($\partial L/\partial W$) 한 번의 행렬곱
- 입력에 대한 그래디언트($\partial L/\partial X$, 이전 레이어로 흘러가는 값) 한 번의 행렬곱

즉 forward가 $2ND$(토큰 수 × 파라미터 수)라면 backward는 $4ND$, 학습 1 스텝(토큰 1개당)은 총 $\approx 6ND$ FLOPs — Chinchilla 등 scaling law 논문에서 쓰는 "$6\times$파라미터수$\times$토큰수" 어림이 이 분해에서 나온다.

# 6. 추론 시 메모리
$$ \text{총 메모리} = \text{모델 가중치} + \text{KV 캐시} + \text{peak activations} $$

- **가중치**: $2VD+12LD^2$개 파라미터 × (정밀도별 바이트 수). BF16이면 파라미터당 2바이트.
- **KV 캐시**: $B\cdot S\cdot(KH)\cdot L\cdot 2$ (K, V 두 개). 자세한 축소 기법은 [[KV Cache Optimization#6. 아키텍처 레벨에서 캐시 자체를 줄이기|KV Cache Optimization]] 참고.
- **Activations**: prefill에서 $O(BNS^2+BSF)$, decode(토큰 1개씩)에서는 $O(BNS+BF)$로 훨씬 작다.
  - 레이어 내 peak activation 구성: 레이어 입력 $BSD$ + $Q,K,V$ 출력 $3BSD$ + attention 행렬 $BNS^2$(FlashAttention 없을 때) + FFN 중간값 $BSF$.
  - [[Long-sequence Handling#4.1. FlashAttention (v1, v2, v3)|FlashAttention]]을 쓰면 $S\times S$ 행렬을 아예 만들지 않으므로 이 항이 $O(S^2)\to O(S)$로 줄어든다.
  - `torch.inference_mode()`는 중간값을 즉시 해제하므로, 실질적으로 중요한 건 레이어 하나의 **peak** 메모리다(레이어 전체를 합산한 값이 아님).

**메모리 지배 요인은 상황마다 다르다.**
- 배치·문맥이 작으면 → 가중치가 지배적.
- prefill에서 문맥이 매우 길면 → (FlashAttention 없이는) $S^2$ activation 항이 지배적.
- 배치가 크면 → KV 캐시와 activation이 함께 커진다.

# 7. 학습 시 메모리
$$ \text{총 메모리} = \text{모델 가중치} + \text{옵티마이저 상태} + \text{그래디언트} + \text{activations} $$

파라미터 수 $P$에 대해, mixed precision(BF16 forward + FP32 master) + Adam을 기준으로:

| 항목 | 설명 | 바이트 |
| :--- | :--- | :--- |
| FP32 master weight | 정밀한 업데이트를 위한 원본 가중치 | $4P$ |
| BF16 forward copy | forward pass 연산용 사본 | $2P$ |
| Adam 1차 모멘트 $m$ (FP32) | optimizer state | $4P$ |
| Adam 2차 모멘트 $v$ (FP32) | optimizer state | $4P$ |
| Gradient (FP32) | BF16으로 계산되어도 누적은 FP32로 | $4P$ |
| **합계** | | $\mathbf{18P}$ |

- 이것이 "mixed-precision Adam은 파라미터당 18바이트"라는 잘 알려진 어림의 근거다.
- **Activations**: 레이어당 $14BSD+BNS^2$ (FlashAttention 없을 때). FlashAttention을 쓰면 두 번째 항이 $BNS$로 줄어, activation 메모리가 시퀀스 길이의 제곱이 아니라 **전체 토큰 수($BS$)에 선형**으로 스케일한다.
- 역전파 때 각 레이어의 activation이 그래디언트 계산에 필요하므로 메모리에 남아 있어야 한다 — 이게 학습에서 activation 메모리가 흔히 병목이 되는 이유이며, **gradient checkpointing**(일부 activation을 버리고 backward 때 재계산)으로 줄일 수 있다.

# 8. 요약 표
| 항목 | 공식 |
| :--- | :--- |
| 파라미터 수 | $2VD+12LD^2$ |
| Forward FLOPs (prefill) | $2BSD(12LD+2LS+V)$ |
| Backward FLOPs | Forward의 약 2배 (총합 $\approx 6ND$) |
| KV 캐시 크기 | $2BSKHL$ |
| 추론 activation (peak, FA 없이) | $O(BNS^2+BSF)$ prefill / $O(BNS+BF)$ decode |
| 학습 메모리 (mixed precision Adam) | $18P$ (가중치+옵티마이저+그래디언트) $+$ activations |

# 9. 한 줄 요약
> 모델 하나의 파라미터·FLOPs·메모리는 전부 "레이어 안의 각 행렬곱 shape을 $2mnp$ 규칙으로 세는 것"으로 환원되며, 학습 메모리는 대략 파라미터당 18바이트(mixed-precision Adam) + activations, 추론 메모리는 가중치 + KV 캐시 + peak activation으로 나뉜다.
