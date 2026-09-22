---
title: Precision
alias: 수치 정밀도 (Mixed Precision Training)
publish: true
date: 2026-09-22
tags:
  - Optimization
  - Inference
---

# 1. 개요

대형 모델을 학습·추론할 때 텐서를 몇 비트로 표현하느냐(정밀도, precision)는 메모리 사용량, 속도, 수치 안정성을 동시에 좌우한다. 같은 실수라도 어떤 dtype으로 저장하느냐에 따라 차지하는 메모리와 표현 가능한 값의 범위가 달라진다.

| dtype | 크기 | 부호/지수/가수 비트 | 비고 |
| :--- | :--- | :--- | :--- |
| **FP32** | 4 bytes | 1 / 8 / 23 | 풀 정밀도, 안전하지만 무겁다 |
| **FP16** | 2 bytes | 1 / 5 / 10 | 가수 비트가 많아 정밀하지만 지수 범위(dynamic range)가 좁다 |
| **BF16** | 2 bytes | 1 / 8 / 7 | FP32와 동일한 지수 범위, 대신 가수 정밀도를 희생 |
| **INT8** | 1 byte | - | 양자화 필요, FP32 대비 1/4 메모리 |
| **INT4** | 0.5 byte | - | INT8보다 더 공격적인 양자화 |

>[!attention] FP16 vs BF16
> **FP16**은 가수(mantissa) 비트가 10개로 많아 표현 정밀도는 높지만, 지수(exponent) 비트가 5개뿐이라 표현 가능한 값의 범위(dynamic range)가 좁다. 학습 중 gradient나 activation 값이 이 범위를 벗어나면 오버플로우/언더플로우가 발생한다.
> **BF16**은 지수 비트가 FP32와 동일하게 8개라 range는 FP32만큼 넓지만, 가수 비트가 7개뿐이라 정밀도는 낮다. 대신 학습이 훨씬 안정적이라 대형 모델 학습의 사실상 표준이 되었다.

>[!tip] 왜 낮은 정밀도로도 학습이 되는가
> 행렬곱(matmul)은 수많은 곱셈-누적의 합이라 개별 원소의 반올림 노이즈에 상당히 관대하다. 반면 activation은 값의 분포가 더 넓고 극단값(outlier)이 잘 생기기 때문에 weight보다 양자화하기 어렵다.

# 2. Mixed Precision Training

>[!abstract] 핵심 철학
> 순전파/역전파의 대부분을 차지하는 행렬곱은 BF16으로 빠르게 계산하고, 정밀도가 중요한 파라미터 업데이트만 FP32로 유지한다.

실무에서 쓰이는 절차는 다음과 같다.

- **master weights는 FP32**로 보관한다.
- forward/backward 계산에는 weight의 **BF16 복사본**을 사용한다.
- **activation도 BF16**으로 계산된다.
- **gradient는 BF16으로 계산**되지만, 파라미터에 누적(accumulate)할 때는 **FP32로 캐스팅**한다.

## 2.1 왜 gradient 누적만 FP32여야 하는가

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

만약 $\theta_t = 1.0$이고 $g_t = 0.0001$이라면, BF16은 0 근처에서는 $0.0001$을 표현할 수 있지만 그 값을 $1.0$에 더한 결과인 $1.0001$은 BF16의 가수 정밀도(7비트)로는 반올림되어 그냥 $1.0$이 되어버린다 — **update가 통째로 소실**된다.

이 문제를 피하기 위해 누적된 gradient만 FP32에 정확히 맞으면 충분하다. 개별 gradient 하나하나가 weight보다 훨씬 작은 경우가 대부분이므로, BF16으로 계산은 하되 파라미터에 반영되는 시점(옵티마이저 스텝)에서만 FP32 정밀도를 쓰는 것이다.

>[!tip] PyTorch 구현 디테일
> `param.grad`는 항상 파라미터(`param`)와 같은 dtype으로 저장된다. 따라서 파라미터 자체를 FP32 master weight로 두면, 개별 gradient 값도 자동으로 FP32로 캐스팅되어 저장된다.

## 2.2 intuition 정리

- 행렬곱은 반올림 노이즈에 강하므로 순전파/역전파 전체를 BF16으로 돌려도 무방하다.
- master weight를 FP32로 두는 이유는 순전파 자체가 아니라, **개별 gradient가 weight에 비해 너무 작다**는 문제 때문이다.

# 3. 모델 로딩과 자동 혼합정밀도 (실전)

## 3.1 HuggingFace에서 BF16으로 로드

```python
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    torch_dtype=torch.bfloat16,  # 권장: weight와 activation 모두 BF16
)
```

- `torch_dtype=torch.bfloat16`: weight와 activation을 모두 BF16으로 로드한다. 추론 시 가장 무난하게 권장되는 선택.
- `load_in_8bit=True`: `LLM.int8()` 알고리즘을 사용해 weight는 INT8, activation은 FP16으로 둔다. **weight 압축을 통한 메모리 절감**이 목적이며, activation이 여전히 FP16이므로 완전한 INT8 연산은 아니다.

## 3.2 이미 로드된 모델을 캐스팅

```python
model = MyModel()
model.load_state_dict(torch.load('model.pt'))
model = model.half()  # 또는 model.to(torch.bfloat16)
```

모든 weight가 BF16이 되고 연산도 BF16으로 수행된다. 실무적으로는 추론 시 모델 전체를 BF16으로 돌리는 것만으로 충분한 경우가 대부분이다.

## 3.3 `torch.autocast`: 연산별 선택적 캐스팅

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(x)
```

`model.half()`와 달리 **weight 자체는 FP32로 유지**하면서, 연산(op) 단위로 필요할 때만 낮은 정밀도를 쓴다.

>[!attention] autocast의 연산별 정밀도 배정
> - **matmul**: 낮은 정밀도(BF16/FP16)를 잘 견디므로 지정한 `dtype`으로 수행
> - **softmax**: 수치 안정성이 중요해 FP32로 수행
> - **layernorm**: 평균/분산 등 reduction 연산이 정밀도에 민감해 FP32로 수행

`autocast`는 모델을 통째로 낮은 정밀도로 바꾸는 것보다 메모리는 더 쓰지만, weight를 FP32로 유지하기 쉽지 않은 기존 코드거나 순수 BF16에서 수치 불안정이 관찰될 때 유용하다.

# 4. 양자화: bitsandbytes로 INT8/INT4 적용

`nn.Linear`를 `bitsandbytes`의 양자화 레이어로 교체하면 weight를 INT8/INT4로 압축할 수 있다.

```python
def replace_linear_with_8bit(model):
    """모든 nn.Linear를 bnb.nn.Linear8bitLt로 교체"""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            new_layer = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
            )
            new_layer.weight = bnb.nn.Int8Params(
                child.weight.data,
                requires_grad=False,
            )
            if child.bias is not None:
                new_layer.bias = nn.Parameter(child.bias.data)
            setattr(model, name, new_layer)
        else:
            replace_linear_with_8bit(child)  # 재귀적으로 하위 모듈 탐색
    return model

model = MyTransformer()
model.load_state_dict(torch.load('model.pt'))
model = replace_linear_with_8bit(model)
model = model.to('cuda')  # 실제 양자화는 CUDA로 옮길 때 일어난다
```

4비트로 더 압축하고 싶다면 `bnb.nn.Linear4bit`로 동일하게 교체하면 된다. 양자화가 실제로 적용되는 시점은 `.to('cuda')` 호출 시점이라는 점에 유의한다.

# 5. 기타: 데이터 로딩

대형 데이터셋을 다룰 때는 `memmap`을 사용하면 전체 데이터를 메모리에 한 번에 올리지 않고 필요한 부분만 디스크에서 읽어올 수 있다. 정밀도 문제와는 별개지만, 대형 모델 학습 파이프라인에서 메모리 예산을 관리하는 또 하나의 축이다.

# 6. 한 줄 요약

> 학습은 **"FP32 master weight + BF16 순전파/역전파 + FP32 gradient 누적"**의 mixed precision으로 안정성과 속도를 모두 잡고, 추론은 **BF16 전체 캐스팅 또는 INT8/INT4 양자화**로 메모리와 속도를 정밀도와 맞바꾼다.
