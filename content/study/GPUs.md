---
title: GPUs
alias: GPU 메모리 계층
publish: true
date: 2026-09-22
tags:
  - Distributed-Training
  - ML
---
# GPUs

## 1. 개요
LLM의 학습/추론 속도를 이해하려면 GPU 자체의 메모리 계층 구조를 알아야 한다. [[Mamba]]의 parallel scan이나 FlashAttention 같은 최적화 기법은 결국 이 계층 구조(느리지만 큰 메모리 vs 빠르지만 작은 메모리)의 격차를 활용하는 것이기 때문이다.

## 2. HBM vs SRAM
### 2.1. HBM (High Bandwidth Memory)
- GPU의 메인 메모리. GPU 연산 유닛 입장에서는 상대적으로 **느린** 메모리다.
- A100 기준 40GB 또는 80GB.
- 모델 파라미터, activation, gradient 등 학습에 필요한 대부분의 데이터가 여기 저장된다.

### 2.2. SRAM
- 칩 내부(on-chip)에 있는 작고 매우 빠른 메모리.
- A100 기준 총 용량은 약 20MB에 불과하다 (HBM 대비 수천 배 작음).

>[!attention] 왜 이 구분이 중요한가
> HBM은 용량은 크지만 대역폭이 상대적으로 병목이 되고, SRAM은 용량은 작지만 접근 속도가 훨씬 빠르다. 연산 자체(compute-bound)보다 **HBM과의 데이터 이동(memory-bound)**이 병목인 경우가 많다.

>[!tip] Kernel Fusion과의 연결
> FlashAttention이나 [[Mamba]]의 Hardware-aware Parallel Scan 같은 기법은 중간 결과를 느린 HBM에 기록했다가 다시 읽어오는 대신, 빠른 SRAM 안에서만 연산을 마치고(kernel fusion) 최종 결과만 HBM에 쓴다. 이를 통해 메모리 대역폭 병목을 제거해 실제 속도를 크게 끌어올린다.

## 3. 한 줄 요약
> GPU는 크지만 느린 HBM과 작지만 빠른 SRAM의 계층 구조를 가지며, 최신 커널 최적화의 핵심은 이 격차를 활용해 중간 연산을 SRAM 안에서 끝내는 것이다.
