---
title: Theoretical CS
alias: 이론 전산학
publish: true
date: 2026-09-22
tags:
  - Math
  - LLM
---
# 1. 개요
최근 LLM/RNN의 표현력(expressivity)을 논할 때 형식 언어 이론(formal language theory)의 언어를 자주 빌려온다. 어떤 아키텍처가 어떤 언어 클래스를 인식할 수 있는지가 곧 그 아키텍처가 어떤 종류의 "알고리즘적 추론"을 할 수 있는지의 척도가 되기 때문이다.

# 2. 정규 언어 (Regular Language)

유한 상태 기계(finite state machine), 즉 **유한 오토마타(finite automaton)**로 인식할 수 있는 언어다. 결정론적 버전을 **DFA (Deterministic Finite Automaton)**라고 부른다.
- 어떤 유한 언어(finite language)든 상태를 충분히 많이 두면 모든 유효한 문자열을 그냥 나열해서 인식할 수 있으므로, **모든 유한 언어는 자명하게 정규 언어**다.

# 3. 문맥무관 언어 (Context-Free Language)

**푸시다운 오토마타(pushdown automaton)** — 유한 상태 기계에 스택(stack) 하나를 더한 것 — 으로 인식할 수 있는 언어다. 스택은 무한한 메모리를 주지만, 그 메모리는 오직 맨 위(top)를 통해서만 접근할 수 있다는 제약이 있다.
- 프로그래밍 언어의 문법(짝이 맞는 괄호, 중첩된 함수 호출, 균형 잡힌 HTML 태그 등)은 전형적으로 문맥무관 언어로 만들어진다.

# 4. DFA를 ReLU RNN으로 인코딩하기

임의의 DFA는 ReLU 활성화를 쓰는 RNN으로 정확히 구현할 수 있다. DFA가 다음처럼 정의되어 있다고 하자.
- 상태 집합 $Q=\{q_1,\dots,q_k\}$
- 알파벳 $\Sigma=\{\sigma_1,\dots,\sigma_m\}$
- 전이 함수 $\delta:Q\times\Sigma\to Q$
- 시작 상태 $q_1$, 수락 상태 $F\subseteq Q$

은닉 차원이 $k$인 RNN을 만들고, 은닉 상태를 현재 DFA 상태의 **one-hot 인코딩**으로 쓴다. 가중치 $W_h\in\mathbb R^{k\times k}$, $W_x\in\mathbb R^{k\times m}$, $b\in\mathbb R^k$를 다음 규칙으로 구성한다.

> 각 전이 $\delta(q_i,\sigma_k)=q_j$에 대해: $(W_h)_{ji}=1$, $(W_x)_{jk}=1$, $b_j=-1$

직관: 상태 $q_i$에 있고(one-hot이라 $h_i=1$, 나머지는 0) 입력 기호 $\sigma_k$를 받으면, $j$번째 pre-activation은 $(W_h h)_j+(W_x x)_j+b_j=1+1-1=1$이 되어 ReLU를 통과한 뒤에도 1로 살아남는다 — 즉 다음 상태 $q_j$의 one-hot이 정확히 재현된다. 이 구성은 DFA가 인식하는 언어(=정규 언어)를 RNN이 정확히 재현할 수 있음을 보여준다.

# 5. 한 줄 요약
> 정규 언어는 유한 오토마타(DFA)로, 문맥무관 언어는 스택이 달린 푸시다운 오토마타로 인식된다. DFA는 은닉 상태를 상태의 one-hot 인코딩으로 쓰는 ReLU RNN으로 정확히 시뮬레이션할 수 있어, RNN의 표현력이 최소한 정규 언어 전체를 포괄한다는 것을 보여주는 구성적 증명이 된다.
