---
title: RAG
alias: Retrieval-Augmented Generation
publish: true
date: 2026-01-14
tags:
  - AI
---
# Retrieval-Augmented Generation (RAG)

## 1. 개요
**RAG**는 LLM이 학습하지 않은 외부 데이터(External Knowledge)를 검색하여 프롬프트에 주입(Augmentation)함으로써, **환각(Hallucination)** 을 줄이고 **최신 정보**를 답변하게 만드는 기술 프레임워크다.

> [!abstract] 핵심 파이프라인
> `Query` $\rightarrow$ `Retrieval` $\rightarrow$ `Prompting` $\rightarrow$ `Generation`
## 2. Retrieval (검색: 핵심 엔진)
RAG의 성능을 결정짓는 가장 중요한 단계. "어떻게 찾을 것인가?"에 따라 전략이 나뉜다.

### 2.1. 검색 기술의 진화 (Series)
아래 기술들은 속도와 정확도의 Trade-off 관계에 있다. 상황에 맞춰 선택하거나 조합(Hybrid)해야 한다.
- **[[Dense Sparse Embedding|1부: Keyword vs Semantic]]**
    - **BM25 (Sparse)**: 고유명사, 키워드 매칭에 강함.
    - **Bi-encoder (Dense)**: 문맥 파악, 의미 검색에 강함.
- **[[Reranking|2부: Reranking & Cross-encoder]]**
    - **Cross-encoder**: 속도는 느리지만 정확도는 최상. 1차 검색 후 검증(Reranking) 용도로 사용.
- **[[Late Interaction|3부: Late Interaction (ColBERT)]]**
    - **MaxSim**: 벡터를 압축하지 않고 토큰 단위로 상호작용하여 속도와 정확도를 모두 잡음.

### 2.2. 인덱싱 전략
- **Chunking**: 문서를 의미 단위로 자르는 전략 (Fixed-size, Recursive, Semantic Chunking).
- **Vector DB**: Pinecone, Milvus, Chroma 등의 저장소 활용.

## 3. Augmentation (증강 & 최적화)
검색된 문서를 LLM이 잘 이해하도록 가공하는 단계.

- **Context Window Management**: LLM의 입력 길이 제한(Context Length)을 고려하여 중요 정보만 남김.
    - *관련 노트*: [[Long-sequence Handling|Long Context Handling]]
- **Prompt Engineering**: "Context를 바탕으로 대답해, 모르면 모른다고 해" 등의 지시문 설계.
## 4. Generation & Safety
최종적으로 답변을 생성하고 검증하는 단계.

- **Generation**: LLM이 검색된 정보를 바탕으로 문장 생성.
    - *관련 노트*: [[Inference Acceleration]] (빠른 생성을 위한 기술)
- **Fallback Strategy**: 검색 결과가 없거나 신뢰도가 낮을 때의 대처법.
    - *관련 노트*: [[AI Fallback Strategy]] (검색 실패 시 대응)
## 5. Advanced RAG Topics
(추후 공부해서 채워넣을 항목들 Seedling 🌱)
- **HyDE (Hypothetical Document Embeddings)**: 가상의 답변을 생성해서 검색에 활용.
- **Modular RAG**: 검색 모듈을 자유롭게 뗐다 붙였다 하는 구조.
- **RAGAS**: RAG 파이프라인의 성능 평가(Evaluation) 프레임워크.