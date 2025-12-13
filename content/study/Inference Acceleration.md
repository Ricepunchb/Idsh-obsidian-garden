---
title: Inference Acceleration
publish: true
date: 2025-12-13
alias: LLM 추론 최적화
---

# Inference Acceleration

## 1. 개요
LLM 서비스의 비용과 사용자 경험을 결정짓는 **추론 속도(Latency)** 와 **처리량(Throughput)** 을 극대화하는 기술들의 집합이다.

> [!abstract] 핵심 목표
> 1.  **TTFT (Time To First Token)**: 첫 글자가 나올 때까지의 시간 $\downarrow$ (반응성)
> 2.  **TPS (Tokens Per Second)**: 초당 생성되는 토큰 수 $\uparrow$ (생성 속도)
> 3.  **VRAM Usage**: 메모리 점유율 $\downarrow$ (비용 절감)

## 2. 가속화 기법의 분류 (Taxonomy)
최적화는 크게 **모델 압축**, **메모리 관리**, **알고리즘 개선** 세 가지 축으로 나뉜다.

```mermaid
%%{init: {'theme':'default'}}%%
graph LR
    Root["Inference Acceleration"]
    
    subgraph "Model Compression"
        Comp1["Quantization<br/> INT4, AWQ"]
        Comp2["Pruning<br/>SparseGPT"]
        Comp3["Distillation<br/>KD"]
    end
    
    subgraph "Memory & System"
        Mem1["PagedAttention<br/>vLLM"]
        Mem2["FlashAttention<br/>Kernel Fusion"]
        Mem3["KV Cache Opt<br/>GQA, MQA"]
    end
    
    subgraph "Algorithmic"
        Algo1["Speculative Decoding<br/> Draft & Verify"]
        Algo2["Parallel Decoding<br/>Medusa"]
    end
    
    Root --> Comp1 & Comp2 & Comp3
    Root --> Mem1 & Mem2 & Mem3
    Root --> Algo1 & Algo2
    
    style Root fill:#212121,stroke:#000000,color:#fff
    style Mem1 fill:#E3F2FD,stroke:#1565C0
    style Algo1 fill:#FFF9C4,stroke:#FBC02D
    style Comp3 fill:#E8F5E9,stroke:#2E7D32
```

### 2.1. 모델 압축 (Model Compression)
모델의 크기 자체를 줄여 연산량과 메모리 접근을 최소화한다.
-   **Quantization (양자화)**: FP16(16bit) 가중치를 INT8, INT4로 줄임. (AWQ, GPTQ)
    -   *관련 노트*: [[QLoRA]] (학습 관점의 양자화지만 원리는 동일)
-   **Pruning (가지치기)**: 중요하지 않은 가중치를 0으로 만들거나 삭제.
-   **Distillation (지식 증류)**: 큰 모델의 지식을 작은 모델로 옮김.
    -   *관련 노트*: [[Knowledge Distillation]]

### 2.2. 메모리 & I/O 최적화 (System Level)
LLM 추론의 병목인 **Memory Bandwidth**를 해결하는 기술이다.
-   **PagedAttention**: OS 페이징 기법으로 KV Cache 단편화 해결.
    -   *관련 노트*: [[PagedAttention|Paged Attention & KV Cache]]
-   **FlashAttention**: GPU SRAM을 활용해 $O(N^2)$ 메모리 접근을 줄임.
    -   *관련 노트*: [[Long-seq|FlashAttention]] (Long Context 노트 참조)

### 2.3. 알고리즘 가속 (Algorithmic Level)
생성 방식(Decoding) 자체를 바꿔 속도를 높인다.
-   **Speculative Decoding**: 작은 모델로 초안을 쓰고 큰 모델이 검수.
    -   *관련 노트*: [[Speculative Decoding]]
-   **Parallel Decoding**: 여러 토큰을 한 번에 생성 (Medusa 등).

## 3. 대표적인 서빙 프레임워크
이 기술들을 집대성하여 실무에서 사용하는 엔진들이다.

| 프레임워크                 | 특징                   | 주요 기술                                  |
| :-------------------- | :------------------- | :------------------------------------- |
| **vLLM**              | 현재 가장 대중적인 고성능 서빙 엔진 | PagedAttention, Continuous Batching    |
| **TensorRT-LLM**      | NVIDIA 최적화, 압도적인 속도  | FP8, In-flight Batching, Kernel Fusion |
| **TGI (HuggingFace)** | 배포 편의성, 생태계 호환성      | FlashAttention, Quantization           |
| **SGLang**            | 복잡한 프롬프트 처리 최적화      | RadixAttention (KV Cache 재사용)          |

## 4. 최적화 전략 가이드
상황에 따라 어떤 기술을 먼저 적용해야 할까?

1.  **메모리가 부족하다?** $\to$ **Quantization (INT4/AWQ)**, GQA
2.  **동시 접속자가 많다?** $\to$ **PagedAttention (vLLM)**, Continuous Batching
3.  **한 명의 응답 속도가 느리다?** $\to$ **Speculative Decoding**, FlashAttention
