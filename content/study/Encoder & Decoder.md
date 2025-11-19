---
title: Encoder Decoder
alias: 인코더 디코더

publish: true

tags:
  - Study
  - AI

date: 2024-07-05
---


# Encoder
RNN, LSTM 등의 seq 처리 모델이 입력된 문장을 끝까지 읽고 마지막으로 내놓은 압축된 정보를 내놓으면, 이 모듈을 Encoder 라고 할 수 있다. 보통 압축된 정보는 Context Vector 라고 함.

# Decoder
Decoder는 Context Vector 를 활용해 출력 seq 를 만든다. Encoder와 비슷하게 RNN, LSTM으로 이루어진 모듈. 이 Context Vector을 $h_0$로, seq 시작을 알리는 스페셜 토큰 "\<SoS\>"를 $x_0$ 으로 받는다. 이 이후는 두 가지로 나뉜다.
1. 각 출력값을 다음 시점의 입력값으로 사용하는 방법
   - 1번째 출력을 2번째의 입력으로, 2번째 출력을 3번째의 입력으로..
   - 출력이 틀려도 일단 입력으로 사용
   - 학습 속도는 느리지만 실제 환경과 가까움

2. Teacher Forcing 방법
   - 각 단계에서 모델 출력 상관없이 정답 seq의 각 토큰을 입력으로 사용
   - 학습 속도는 빠르지만 실제 환경과 거리가 멈

# 단점
seq 길이가 길수록 시작 지점의 정보는 gradient decay 때문에 참조가 안될 수 있음
LSTM을 써도 long-term dependency 문제를 완벽하게 극복은 못 함
[[Attention 메커니즘]] 이 나온 이유!