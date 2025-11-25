---
title: Continual Training
publish: true
date: 2025-11-25
tags:
  - AI
  - LLM
---
# Continual Learning 파이프라인

> LLM을 실제 서비스까지 끌고 가는 데 가장 중요한 두 단계

## 1. Continual Pre-training (CPT)
이미 잘 학습된 base model에 **새로운 지식**을 계속 주입하는 단계  
→ "모델의 세계관을 계속 업데이트한다"는 느낌

### 왜 하는가?
- 도메인 적응 (의학, 법률, 금융, 수학 등 전문 지식)
- Knowledge cutoff 이후의 최신 정보 반영 (2024~2025년 논문, 뉴스, 코드 등)
- 새로운 언어 추가 (예: Llama 3 8B → 한국어/일본어 고품질 데이터로 CPT)
- 기업 내부 문서/코드베이스 주입

### 주로 쓰는 데이터
- 수십~수백 TB 규모의 **unlabeled raw text**
- arXiv 최신 논문, 책, 위키, 기업 내부 문서, 코드 등
- 고품질 필터링 + deduplication 필수

### 특징 및 주의점
- From-scratch보다 **10~50배 저렴** (파라미터 효율적 초기화 덕분)
- Catastrophic Forgetting 발생 가능 → 최근엔 거의 안 일어남
  - LoRA/QLoRA만 써도 충분히 방어 가능
  - 필요하면 Replay (예전 데이터 0.5~2% 섞기) 또는 EWC, Lambda 등 정규화 기법
- 2025년 기준 실무에선 **8B~70B 모델에 1~10T 토큰 추가 CPT** 하는 게 표준

## 2. Continual Post-training (Alignment & Capability Refinement)
지식 주입이 끝난 후, 모델이 **잘 대답하게 만드는 단계**  
→ SFT + RLHF + DPO + Iterative 자기 개선 반복

### 왜 하는가?
- Instruction following 능력 극대화
- 수학, 코딩, 논리 추론 등 **특정 능력 전문화**
- 안전성/윤리 가이드라인 지속 업데이트
- 출력 포맷 강제 (JSON, function calling, chain-of-thought 등)
- 사용자 피드백 실시간 반영 (예: Grok의 실시간 RL)

### 주로 쓰는 데이터
- 고품질 **SFT 데이터** (QA pairs, chain-of-thought, tool-use 등)
- Preference 데이터 (RLHF/DPO/KTO)
- Synthetic 데이터 (Self-Instruct, Evol-Instruct, MCTS, STaR, Quiet-STaR 등)
- 검증된 self-generated 데이터 (Llama 3, Qwen2, DeepSeek-R1 방식)

### 2025년 최신 트렌드
- **Iterative Post-training** 루프가 대세
  - 모델이 스스로 데이터 생성 → 검증 → 다시 학습 → 반복
  - Llama 3, Qwen2, DeepSeek-Math, AIME 2025 1등 모델들 모두 이 방식
- Pure RLHF는 거의 안 쓴다 (DPO, KTO, ORPO 등으로 대체)
- Reasoning 모델은 **Process Reward Model (PRM)** + **Outcome Reward Model (ORM)** 병행

### CPT vs Post-training 한눈에 비교

| 구분                | Continual Pre-training          | Continual Post-training                  |
|---------------------|----------------------------------|-------------------------------------------|
| 주 목적             | 지식 확장                        | 능력/행동 정교화                           |
| 데이터 타입          | Unlabeled raw text               | Labeled / Preference / Synthetic          |
| 주요 기법            | LoRA + 긴 Context 계속 학습       | SFT → DPO/ORPO → Iterative self-refine   |
| Catastrophic Forgetting | 거의 없음 (LoRA 덕분)           | 거의 없음 (Preference 학습이라 안전)       |
| 비용                 | 상대적으로 저렴                   | 데이터 품질에 따라 매우 비쌈               |
| 실무 예시            | Llama 3 한국어 CPT, 기업 내부 지식 주입 | Grok 실시간 RL, DeepSeek-R1 수학 특화     |

### 한 줄 요약
> CPT는 “모델에게 새로운 책을 읽혀주는 과정”  
> Post-training은 “그 책 내용을 얼마나 잘 설명하고, 예의 바르게 말하게 만들지 반복 첨삭하는 과정”
