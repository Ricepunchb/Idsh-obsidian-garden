---
title: Agent AI 시스템 아키텍처 패턴 설계
alias: AI Agent 5부
publish: true
date: 2026-05-20
tags:
  - AI
  - Agent
---
연관 노트: [[AI Agent의 탄생]], [[Agent는 어떻게 생각하는가]], [[Agent는 어떻게 행동하는가]], [[Multi-Agent 시스템]]

실무에서 AI Agent 시스템을 구축할 때 정확히 어떤 구조를 짜야 할까?

모든 문제에 만능인 디자인은 없다.

작업의 복잡도, 요구되는 응답 속도, 비용, 인간의 개입 여부에 따라 적절한 디자인 패턴을 선택해야 한다

실무에서 널리 쓰이는 아키텍쳐 패턴들을 성격에 따라 분류하고 정리해 본다.

# 1. Single-agent vs Multi-agent
가장 첫 번째 갈림길은 에이전트의 '수'이다.

- **Single-agent (단일 에이전트)**: 하나의 LLM이 목표를 이해하고, 도구를 선택하며, 작업을 완수합니다. 개발 초기 프로토타입이나, 비교적 직선적인 다단계 작업(예: DB 조회 후 뉴스 요약)에 적합합니다.
    ![[choose-design-pattern-agentic-ai-system-single-agent.svg]]
- **Multi-agent (다중 에이전트)**: 단일 에이전트가 처리하기 벅찬 복잡한 문제를 여러 개의 **'전문 에이전트(Specialized Agents)'** 로 나누어 해결합니다.

# 2. Multi-agent 패턴
## Deterministic
AI 모델이 작업 순서를 고민하지 않고, **사전에 정의된 명확한 파이프라인**을 따라 움직이는 구조입니다. 오케스트레이션(Orchestration)에 LLM을 쓰지 않아 비용과 지연 시간을 줄일 수 있습니다.

- **Sequential (순차 패턴)**: 이전 에이전트의 결과물이 다음 에이전트의 입력으로 들어가는 선형 구조입니다. (예: 데이터 추출 $\rightarrow$ 정제 $\rightarrow$ DB 적재)
    ![[choose-design-pattern-agentic-ai-system-sequential.svg]]

- **Parallel (병렬 패턴)**: 하나의 입력을 여러 에이전트가 동시에 처리한 후 결과를 취합(Gather)합니다. 전체 지연 시간을 크게 줄일 수 있습니다.
  ![[choose-design-pattern-agentic-ai-system-parallel.svg]]

- **Review & Critique (검토 및 비평)**: 생성 에이전트가 초안을 만들면, 평가 에이전트가 기준(보안, 포맷 등)에 따라 검증하고 피드백을 줍니다.
    ![[choose-design-pattern-agentic-ai-system-review-critique.svg]]

- **Iterative Refinement (반복 개선)**: 목표 퀄리티에 도달할 때까지 작업을 계속 수정하고 다듬습니다. 코딩이나 긴 글쓰기에 적합합니다.
  ![[choose-design-pattern-agentic-ai-system-iterative-refinement.svg]]

## Dynamic
정해진 길이 아니라, AI 모델 스스로 상황을 판단하여 **어떤 에이전트에게 일을 맡길지 동적으로 결정**하는 패턴입니다.

- **Coordinator (조정자 패턴)**: 중앙의 '조정자'가 사용자의 요청을 분석하고, 가장 적합한 전문 에이전트에게 작업을 라우팅합니다. 유연하지만 비용이 증가합니다.
    ![[choose-design-pattern-agentic-ai-system-coordinator.svg]]

- **Hierarchical (계층형 작업 분할 패턴)**: 최고 관리자(Root)가 거대한 문제를 하위 작업으로 쪼개어 중간 관리자에게 넘기는 트리(Tree) 구조입니다. 거대한 리서치 업무에 적합합니다.
    ![[choose-design-pattern-agentic-ai-system-hierarchical-task.svg]]

- **Swarm (군집 패턴)**: 중앙 통제 없이, 전문 에이전트들이 서로 자유롭게 의견을 주고받으며 해결책을 합의(Consensus)해 나갑니다. 창의적인 작업에 유리하나 제어가 까다롭습니다.
  ![[choose-design-pattern-agentic-ai-system-swarm.svg]]

# 3. 특수 목적 패턴
- **ReAct(Reason and Act)**: 특정 end 조건을 만족할때까지 '생각 $\rightarrow$ 행동 $\rightarrow$ 관찰'의 루프를 돈다. 끊임없이 변하는 동적환경에 적합
  ![[choose-design-pattern-agentic-ai-system-react.svg]]

- **Human-in-the-loop(인간 개입 패턴)**: 보안이나 금융 등 중대한 결정을 내려야 할 때 시스템이 일시 정지하고 **인간의 승인(Approval)** 을 기다립니다. 신뢰성을 담보하는 필수 패턴
  ![[choose-design-pattern-agentic-ai-system-human-in-the-loop.svg]]

- **Custom Logic (맞춤 로직 패턴)**: 위의 패턴들을 섞거나, 코드 수준의 `If-Else` 조건문 등을 하드코딩하여 완전히 독자적인 워크플로우를 구축합니다.