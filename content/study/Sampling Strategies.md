---
title: Sampling Strategies
alias: 디코딩 샘플링 전략
publish: true
date: 2026-09-22
tags:
  - Inference
  - LLM
---
# 1. 개요
LLM은 다음 토큰에 대한 확률 분포(logits → softmax)를 뱉어줄 뿐, "어떤 토큰을 실제로 고를지"는 별도의 샘플링 전략이 결정한다. 항상 1등만 뽑으면(greedy) 답이 뻔하고 반복적이 되고, 무작위로 뽑으면 헛소리가 늘어난다 — 그 사이 어딘가를 조절하는 것이 이 노트의 주제다.

> [!abstract] 핵심 철학
> **"분포를 얼마나 뾰족하게(확신에 차게) 만들지, 그리고 그 분포의 어디까지를 후보로 남길지"를 조절하는 것.**

# 2. Temperature
softmax에 들어가기 전 logits를 $T$로 나눈다.
$$ p_i = \frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)} $$
- $T\to 0$: 분포가 극단적으로 뾰족해져 사실상 greedy decoding(가장 확률 높은 토큰만 선택)에 수렴. 결정적이고 일관되지만 반복/지루함이 늘어남.
- $T=1$: 원래 모델이 학습한 분포 그대로.
- $T>1$: 분포가 평평해져 낮은 확률의 토큰도 뽑힐 기회가 늘어남 → 창의적이지만 헛소리 위험 증가.
- $T\to\infty$: 완전 균등분포(uniform)에 수렴 — 사실상 무작위 선택.

>[!tip] 직관
> Temperature는 "얼마나 확신에 차서 답할지"를 조절하는 다이얼이다. 코드 생성처럼 정답이 뚜렷한 작업엔 낮게, 창작처럼 다양성이 필요한 작업엔 높게 잡는다.

# 3. Top-k Sampling
확률 상위 $k$개 토큰만 후보로 남기고, 나머지는 확률을 $-\infty$(logit 기준)로 만들어 완전히 배제한 뒤 그 안에서 다시 정규화해 샘플링한다.
- $k$가 작을수록 안전하지만 다양성이 줄고, $k$가 크면 다양하지만 이상한 토큰이 섞일 위험이 커진다.
- **한계**: $k$가 고정값이기 때문에, 분포가 이미 뾰족한 상황(사실상 후보가 2~3개뿐)에서도 항상 $k$개를 강제로 채우거나, 분포가 평평한 상황(후보가 많아야 하는 상황)에서도 $k$개로 제한해버리는 비효율이 있다.

# 4. Top-p (Nucleus) Sampling
누적 확률이 임계값 $p$를 넘는 지점까지만 후보로 남긴다. 확률 내림차순으로 정렬한 뒤,
$$ \text{후보 집합} = \arg\min_{V'\subseteq V} \left\{ |V'| : \sum_{i\in V'} p_i \ge p \right\} $$
가 되는 최소 집합만 남기고 나머지는 배제한다.
- top-k와 달리 **후보 개수가 상황에 따라 자동으로 늘거나 줄어든다** — 분포가 뾰족하면 후보가 2~3개로 줄고, 평평하면 후보가 수십 개로 늘어난다. 그래서 실무에서는 top-k보다 top-p(보통 $p=0.9\sim0.95$)가 더 널리 쓰인다.

# 5. 구현 (의사코드)
```python
def sample(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature

    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, indices, values)

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 누적 확률이 top_p를 넘는 토큰부터 제거
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False  # 1등 토큰은 항상 남긴다

        indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```
- temperature → top-k → top-p 순서로 적용하는 것이 일반적: 먼저 분포의 뾰족함을 조절하고, 그다음 후보군을 좁혀나간다.
- top-k와 top-p는 함께 쓸 수 있다 (top-k로 큰 틀에서 거른 뒤 top-p로 다시 다듬는 식).

# 6. 다른 노트와의 관계
- [[Speculative Decoding]]: draft/target 모델의 accept-reject 알고리즘도 결국 두 분포($p$, $q$) 사이의 샘플링 문제이며, 여기서 다룬 temperature/top-k/top-p로 조정된 분포 위에서 동작한다.
- [[Inference Acceleration]]: 샘플링 전략 자체는 속도보다 품질/다양성 조절이 목적이지만, batching 환경에서는 각 요청이 서로 다른 temperature/top-p를 요구할 수 있어 서빙 엔진 구현에 영향을 준다.

# 7. 한 줄 요약
> Temperature는 분포를 "얼마나 뾰족하게" 만들지, Top-k/Top-p는 그 분포의 "어디까지를 후보로 남길지"를 결정한다 — 특히 Top-p는 후보 개수를 상황에 맞게 자동 조절해 실무 표준이 되었다.
