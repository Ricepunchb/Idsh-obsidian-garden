---
title: Gaussian Mixture Model
alias: 가우시안 혼합 모델 (GMM & EM Algorithm)
publish: true
date: 2026-10-21
tags:
  - Math
  - Probability
  - ML
  - Classical-ML
  - Generative-Model
---
# Gaussian Mixture Model (가우시안 혼합 모델, GMM)

> [!abstract] 핵심 개념 & 3줄 요약
> $$p(x \mid \Theta) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k) \quad \text{where } \sum_{k=1}^{K} \pi_k = 1, \; \pi_k \ge 0$$
> - **한 줄 정의**: 봉우리가 딱 하나뿐(Unimodal)인 단일 [[Gaussian Distribution|가우시안 분포]]의 한계를 넘어, **$K$개의 서로 다른 가우시안 분포를 가중합(선형 결합)으로 포개어** 세상의 아무리 복잡한 다봉(Multi-modal) 확률 분포도 원하는 정밀도로 근사해 내는 **잠재 변수 확률 생성 모델(Latent Variable Generative Model)** 이다.
> - **학습 알고리즘 (EM)**: 로그 안에 덧셈이 들어간 $\log \sum_{k=1}^K$ 때문에 미분해서 0이 되는 닫힌 해가 없다. 대신 **E-Step(현재 파라미터로 각 데이터가 어느 봉우리 출신인지 사후확률 $\gamma_{nk}$ 추론)** 과 **M-Step(추론된 확률을 가중치 삼아 $\mu_k, \Sigma_k, \pi_k$ 갱신)** 을 번갈아 돌리는 **EM(Expectation-Maximization) 알고리즘**으로 푼다.
> - **확장성과 연결고리**: 공분산을 $\Sigma_k = \epsilon I (\epsilon \to 0)$로 줄이면 정확히 **[[군집 분석|K-means 알고리즘]]** 으로 수렴하며, 파라미터 $(\pi_k, \mu_k, \Sigma_k)$를 신경망이 출력하게 만들면 **[[World Model]]** 의 핵심 부품인 **MDN(Mixture Density Network)** 이 된다.

---

## 1. 개요: 왜 단일 가우시안으로는 부족한가?

현실의 데이터는 봉우리가 하나인 예쁜 종 모양(Unimodal)이 아닌 경우가 훨씬 많다.
- **예시 1 (군집화)**: 남성과 여성의 키-몸무게 데이터를 한데 섞어 놓으면, 두 개의 타원형 구름(쌍봉 분포, Bimodal Distribution)이 나타난다. 여기에 단일 가우시안 $\mathcal{N}(\mu, \Sigma)$ 하나를 피팅하면 두 구름 한가운데의 텅 빈 허공을 평균으로 잡는 참사가 벌어진다.
- **예시 2 (로보틱스 / [[VLA]] / [[World Model]])**: 앞에 장애물이 있을 때 왼쪽으로 피하는 궤적($a = -10$)과 오른쪽으로 피하는 궤적($a = +10$)이 공존한다.

> [!tip] 보편 근사 정리 (Universal Approximator of Densities)
> 신경망이 임의의 연속 함수를 근사할 수 있듯, **가우시안 혼합 모델(GMM)은 컴포넌트 개수 $K$만 충분히 늘리면 세상의 어떤 매끄러운 확률 밀도 함수 $p(x)$라도 임의의 정밀도로 근사할 수 있다!**

---

## 2. 확률 생성 과정과 잠재 변수 (Latent Variable $z$)

GMM을 가장 깔끔하게 이해하는 방법은 **"데이터 $x_n \in \mathbb{R}^D$가 어떻게 태어났는가?"** 를 설명하는 2단계 생성 스토리(Generative Process)로 바라보는 것이다.

우리 눈에는 데이터 $x_n$만 보이지만, 사실 보이지 않는 **잠재 변수(Latent Variable / Hidden Variable) $z_n \in \{0, 1\}^K$** (크기 $K$인 원-핫 벡터, $\sum_{k=1}^K z_{nk} = 1$)가 뒤에 숨어 있다고 가정한다.

1. **Step 1 (주사위 굴리기 — 어느 군집에서 뽑을까?)**:
   각 면이 나올 확률이 $\boldsymbol{\pi} = (\pi_1, \dots, \pi_K)$인 $K$면체 주사위를 굴려 군집 번호 $k$를 고른다 (범주형 분포, Categorical Distribution).
   $$p(z_{nk} = 1) = \pi_k \iff p(z_n) = \prod_{k=1}^{K} \pi_k^{z_{nk}}$$
2. **Step 2 (가우시안 샘플링 — 해당 군집에서 점 찍기)**:
   $k$번째 군집이 당첨되었다면($z_{nk}=1$), 그 군집 전용 가우시안 분포 $\mathcal{N}(\mu_k, \Sigma_k)$에서 실제 데이터 $x_n$을 뽑는다.
   $$p(x_n \mid z_{nk} = 1) = \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \iff p(x_n \mid z_n) = \prod_{k=1}^{K} \mathcal{N}(x_n \mid \mu_k, \Sigma_k)^{z_{nk}}$$
3. **주변화 (Marginalization)**:
   잠재 변수 $z_n$에 대해 가능한 모든 경우($k=1, \dots, K$)를 다 더하면(Sum Rule), 우리가 아는 GMM의 주변 확률 밀도 함수가 유도된다!
   $$p(x_n) = \sum_{z_n} p(z_n) p(x_n \mid z_n) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)$$

---

## 3. 왜 그냥 미분(MLE)으로 못 풀고 EM 알고리즘을 쓰는가?

$N$개의 독립 데이터 $X = \{x_1, \dots, x_N\}$이 주어졌을 때, 파라미터 $\Theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$를 찾기 위해 **로그 우도(Log-Likelihood)** 를 적어보자.

$$\log p(X \mid \Theta) = \sum_{n=1}^{N} \log p(x_n \mid \Theta) = \sum_{n=1}^{N} \underbrace{\log \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \right)}_{\text{문제의 원흉: }\log \text{ 안에 } \sum \text{이 들어있음!}}$$

- **단일 가우시안($K=1$)일 때**: $\log$와 가우시안의 $\exp\big(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\big)$가 곧바로 만나 서로 상쇄되면서 깔끔한 2차 다항식이 되어, 미분해서 0으로 놓으면 단번에 $\mu = \frac{1}{N}\sum x_n$이 풀린다.
- **GMM($K \ge 2$)일 때**: **$\log$와 $\exp$ 사이에 덧셈 기호 $\sum_{k=1}^K$가 가로막고 있어서 $\log$가 안으로 파고들지 못한다!** 이 상태에서 $\mu_k$로 편미분해 보면 미지수 $\mu_k$가 지수 안에도 있고 분수 밖에도 서로 얽혀 있어서 닫힌 해(Closed-form Solution)가 존재하지 않는다.

---

## 4. 수식 완벽 유도: EM 알고리즘 (ELBO와 Responsibility)

이 난제를 푸는 열쇠는 **젠센 부등식(Jensen's Inequality: $\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$)** 을 이용해 $\sum_{k=1}^K$를 $\log$ 바깥으로 끌어내는 것이다. (이 과정은 [[VAE]]의 ELBO 유도 과정과 수학적으로 완전히 동일하다!)

### 4.1. ELBO (Evidence Lower Bound)와 E-Step의 탄생
임의의 잠재 변수 확률 분포 $q(z_n)$을 도입해 로그 우도를 변형하자.
$$\begin{aligned}
\log p(x_n \mid \Theta) &= \log \sum_{k=1}^{K} q(z_{nk}=1) \frac{p(x_n, z_{nk}=1 \mid \Theta)}{q(z_{nk}=1)} \\
&\ge \sum_{k=1}^{K} q(z_{nk}=1) \log \frac{p(x_n, z_{nk}=1 \mid \Theta)}{q(z_{nk}=1)} \equiv \mathcal{L}(q, \Theta) \quad (\text{ELBO})
\end{aligned}$$
여기서 부등호($\ge$)가 **정확히 등호($=$)가 되어 하한선이 최대화되는 조건**은, 보조 분포 $q(z_{nk}=1)$가 현재 파라미터 $\Theta^{(t)}$ 하에서의 **진짜 사후 확률(Posterior Probability) $p(z_{nk}=1 \mid x_n, \Theta^{(t)})$** 와 일치할 때다!

베이즈 정리(Bayes' Rule)에 의해 이 사후 확률을 **책임값(Responsibility, $\gamma_{nk}$)** 이라 부른다.

> [!important] 1단계: E-Step (Expectation Step) — "누가 어느 군집 소속인지 확률 계산"
> 현재 파라미터 $(\pi_k, \mu_k, \Sigma_k)$를 고정하고, 데이터 $x_n$이 $k$번째 가우시안에서 나왔을 사후 확률 $\gamma_{nk} \in [0, 1]$를 계산한다.
> $$\gamma_{nk} \equiv p(z_{nk} = 1 \mid x_n, \Theta) = \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}$$

---

### 4.2. M-Step의 닫힌 해 유도
이제 $q(z_{nk}=1) = \gamma_{nk}$로 고정하면, $\sum_{k=1}^K$가 $\log$ 밖으로 빠져나온 기대 완비 데이터 로그 우도(Expected Complete-Data Log-Likelihood) $Q(\Theta, \Theta^{(t)})$를 얻는다.

$$Q(\Theta, \Theta^{(t)}) = \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{nk} \Big( \log \pi_k + \log \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \Big)$$
$$= \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{nk} \left( \log \pi_k - \frac{1}{2}\log |\Sigma_k| - \frac{1}{2}(x_n - \mu_k)^\top \Sigma_k^{-1} (x_n - \mu_k) + \text{const} \right)$$

보라! 드디어 **$\log$와 $\exp$가 직접 만나 지수가 다 풀렸다!** 이제 각 파라미터로 편미분하여 0으로 놓으면 너무나 직관적인 가중 평균 형태의 업데이트 공식이 튀어나온다. 먼저 $k$번째 군집에 할당된 **유효 데이터 개수(Effective Number of Points)** 를 $N_k = \sum_{n=1}^{N} \gamma_{nk}$라 정의하자.

> [!important] 2단계: M-Step (Maximization Step) — "책임값($\gamma_{nk}$)을 가중치로 파라미터 업데이트"
> 1. **평균 벡터 $\mu_k^{\text{new}}$ 업데이트** ($\nabla_{\mu_k} Q = 0$):
>    $$\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk} \, x_n$$
> 2. **공분산 행렬 $\Sigma_k^{\text{new}}$ 업데이트** ($\nabla_{\Sigma_k^{-1}} Q = 0$):
>    $$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk} (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^\top$$
> 3. **혼합 계수 $\pi_k^{\text{new}}$ 업데이트** ($\sum_{k=1}^K \pi_k = 1$ 제약 하에 라그랑주 승수법 적용):
>    $$\pi_k^{\text{new}} = \frac{N_k}{N} = \frac{1}{N} \sum_{n=1}^{N} \gamma_{nk}$$

---

## 5. 수학적 증명: K-means는 GMM의 극한($\epsilon \to 0$)이다!

[[군집 분석]]에서 다루는 **K-means 알고리즘**은 사실 GMM에서 아주 특별한 가정을 추가한 극한 케이스다. 이를 수식으로 직접 확인해보자.

모든 군집의 공분산 행렬이 **반지름 $\sqrt{\epsilon}$인 구형(Isotropic) 행렬 $\Sigma_k = \epsilon I$** 로 똑같고 고정되어 있다고 가정하자. 이때 가우시안 확률 밀도는 다음과 같다.
$$\mathcal{N}(x_n \mid \mu_k, \epsilon I) = \frac{1}{(2\pi \epsilon)^{D/2}} \exp\left( -\frac{\| x_n - \mu_k \|^2}{2\epsilon} \right)$$

이를 E-Step의 책임값 $\gamma_{nk}$ 수식에 대입하자:
$$\gamma_{nk} = \frac{\pi_k \exp\left( -\| x_n - \mu_k \|^2 / 2\epsilon \right)}{\sum_{j=1}^{K} \pi_j \exp\left( -\| x_n - \mu_j \|^2 / 2\epsilon \right)}$$

이제 **분산 $\epsilon \to 0^+$ (공의 크기를 점으로 수축!) 극한**을 보내면 무슨 일이 일어날까?
- 분모의 합 $\sum_{j=1}^K$ 중에서 **거리 $\|x_n - \mu_j\|^2$가 가장 작은(가장 가까운) 중심점 $j^*$의 지수항**이 다른 모든 항들에 비해 무한히 빠르게(지수적으로) 지배하게 된다!
- 따라서 $\epsilon \to 0$ 극한에서 Soft 확률 $\gamma_{nk}$는 가장 가까운 군집에만 1을 주고 나머지는 전부 0이 되는 **Hard Assignment (K-means 할당 규칙)** 으로 수렴한다!
$$\lim_{\epsilon \to 0^+} \gamma_{nk} = r_{nk} = \begin{cases} 1 & \text{if } k = \arg\min_j \| x_n - \mu_j \|^2 \\ 0 & \text{otherwise} \end{cases}$$
- 동시에 ELBO 목적함수는 정확히 K-means의 **군집 내 분산(WCV) 최소화 수식 $-\sum_n \sum_k r_{nk} \|x_n - \mu_k\|^2$** 으로 귀결된다.

| 비교 항목 | **K-means (Hard Clustering)** | **GMM (Soft Clustering)** |
| :--- | :--- | :--- |
| **할당 방식** | $\gamma_{nk} \in \{0, 1\}$ (무조건 한 군집에만 배정) | **$\gamma_{nk} \in [0, 1]$ (소속 확률 분포 제공)** |
| **군집 모양 가정** | 모든 군집이 **반지름이 똑같은 동그란 구(Sphere)** | **각 군집마다 크기도 다르고 비스듬히 찌그러진 타원체($\Sigma_k$) 가능** |
| **수학적 정체** | $\Sigma_k = \epsilon I, \, \epsilon \to 0$ 극한의 특수해 | 일반화된 확률 생성 모델 (밀도 추정 및 샘플 생성 가능) |

---

## 6. 딥러닝과의 결합: Mixture Density Network (MDN)

1994년 Christopher Bishop이 제안한 **MDN(Mixture Density Network)** 은 신경망(MLP/RNN)과 GMM을 합친 모델이다.
- 일반 신경망이 입력 $x$를 받아 단일 출력 $\hat{y} = f_\theta(x)$를 뱉는 대신, **신경망이 $K$개 가우시안 혼합 분포의 파라미터 $(\pi_k(x), \mu_k(x), \sigma_k(x))_{k=1}^K$를 직접 출력**한다.
$$p(y \mid x; \theta) = \sum_{k=1}^{K} \pi_k(x; \theta) \, \mathcal{N}\big( y \mid \mu_k(x; \theta), \, \sigma_k^2(x; \theta) I \big)$$
- **어디에 쓰이는가?**:
  1. **[[World Model]] (Ha & Schmidhuber, 2018)**: 미래의 상태 $z_{t+1}$은 결정론적이지 않고 여러 갈래로 나뉠 수 있다(예: 적 캐릭터가 왼쪽으로 튈지 오른쪽으로 튈지 모름). 그래서 RNN 뒤에 MDN을 붙인 **MDN-RNN**으로 미래 잠재 상태의 다봉 확률 분포를 예측했다!
  2. **로봇 모방학습 (Behavior Cloning with GMM Head)**: Diffusion Policy가 등장하기 전(RoboMimic 등)에 로봇 행동의 다중 모달리티(장애물 좌/우 회피)를 해결하기 위해 정책 네트워크의 출력 헤드로 가장 널리 쓰였다.

---

## 7. GMM의 주의점: 특이점 붕괴 (Singularity Collapse)

GMM을 최대우도추정(EM)으로 그냥 학습시키면 가끔 손실이 $-\infty$로 발산하며 터지는 경우가 있다.
- **원인**: 만약 $k$번째 가우시안의 중심 $\mu_k$가 운 나쁘게 **특정 데이터 포인트 $x_n$ 하나 위에 정확히 겹쳐 앉아버리면($\mu_k = x_n$)**, 그 군집이 다른 데이터는 다 무시하고 자기 분산 $\sigma_k \to 0$으로 줄여버릴 때 가우시안 높이가 $\frac{1}{\sqrt{2\pi}\sigma_k} \to \infty$로 치솟아 전체 우도가 무한대가 된다!
- **해결책**: 공분산 행렬 대각선에 아주 작은 양수 정규화 항을 강제로 더해주거나($\Sigma_k \leftarrow \Sigma_k + \lambda I$, 보통 $\lambda = 10^{-6}$), 공분산에 역위샤트 사전분포(Inverse-Wishart Prior)를 걸어주는 **Variational GMM**을 사용한다.

---

## 8. 한 줄 요약

> **Gaussian Mixture Model = "단일 가우시안 여러 개를 포개어 세상의 모든 복잡한 다봉 분포를 표현하는 생성 모델이자, EM 알고리즘으로 학습되며 극한에서는 K-means를, 신경망과 만나면 월드 모델의 MDN을 낳는 확률 모델링의 중추"**

---
연관 노트: [[Gaussian Distribution]], [[군집 분석]], [[World Model]], [[VAE]], [[VLA]]

