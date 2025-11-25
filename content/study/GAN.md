---
title: GAN
alias: Generative Adversarial Network
publish: true
tags:
  - AI
  - Model
date: 2024-07-05
---



# 개요
- 2인 게임 상황을 가정하여 새로운 데이터를 생성하는 모델
- Generator : 실제와 유사한 데이터를 위조하여 만듦
- Discriminator : Generator가 위조한 데이터와 실제 데이터 중 뭐가 실제인지 판별
- 위 둘은 서로 적대적 관계로 게임 환경 속에서 서로 성장하는 구조

# Training
>[!note]notation
>$x$ : 실제데이터
>$z$ : 임의의 노이즈
>$D(x)$ : 입력 $x$에 대한 Discriminator의 판단 $(0 \sim 1)$
>$G(z)$ : 입력 $z$에 대한 Generator에 의해 생성되는 데이터
>$L_D$ : Discriminator 손실함수
>$L_G$ : Generator 손실함수

$$L_D = -\log D(x)-\log\left(1-D\left(G(z)\right)\right)$$
$D(x)\to1,\quad D(G(z))\to0$일수록 손실 감소

$$L_G = \log(1-D(G(z)))$$
$D(G(z))\to1$일수록 손실 감소

- Generator, Discriminator를 번갈아 가며 Training 진행
- 한쪽이라도 Loss가 작아지면 학습 종료
- 그래서 균형을 맞춰 학습하기가 어렵다

# 문제점
- 손실이 서로 Monotone 하게 감소하는게 아니라, 전체적으로 Oscillation을 그림
- 경쟁구조이다보니 둘 다 손실이 낮을수가 없음
- 다양성부족, Mode collapse
- Generator가 한가지 유형의 데이터만 생성할수도 있음
- $x$의 데이터 distribution이 충분히 고려되지 않아 발생함

# 파생형
위 문제점을 해결하기 위해 여러가지 파생형이 나옴

## cGAN
일반 GAN은 모든 데이터를 골고루 생성하지만, cGAN은 특정조건이나 Label에 해당하는 데이터만을 생성
예) 0~9 를 모두 생성하는 대신 8만 생성

## Style Transfer
Image-to-Image Translation
- 하나의 이미지를 다른 형식으로 변형해줌 예) 얼룩말 사진을 말 사진으로
- cGAN 형태로 구현 가능

## Cycle GAN
- Unpaired 이미지들을 이용해 Style Transfer 구현

## SRGAN
- Super-Resolution : 저해상도를 고해상도로
- 이미지 자체를 조건으로 주어 생성하는 cGAN의 형태