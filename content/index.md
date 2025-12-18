---
title: Home
publish: True

# tags:
#   - example-tag
---
---
# Welcome to RicePunchb's Digital Garden

> [!quote] "The mind is not a vessel to be filled, but a fire to be kindled."
> 이곳은 **AI Research Engineer**로서의 학습 과정과 기술적인 고민들이 자라나는 **디지털 정원**입니다. 단순한 지식의 나열보다는, **근본 원리(Math)**에서 시작하여 **최신 응용(State-of-the-Art)**으로 뻗어나가는 사고의 흐름을 기록합니다.

---
## 🗺️ Map of Content 

### Advanced LLM & Optimization
최신 거대 언어 모델의 학습부터 추론 최적화까지, 엔지니어링의 정점을 다룹니다.

- **Training & Fine-tuning**
    - [[study/Training LLM|🔥 Hands-on LLM Training]] : SFT + QLoRA 실습 파이프라인.
    - [[study/LoRA|LoRA]] & [[study/Instruction Tuning|Instruction Tuning]] : 효율적인 미세조정 기법.
    - [[study/Continual Training|Continual Training]] : 지식 주입(CPT)과 행동 교정의 루프.
    - [[study/Model Merging|Model Merging]] & [[study/Model Upscaling|Model Upscaling]] : 모델을 합치고 키우는 연금술.
    - [[study/Knowledge Distillation|Knowledge Distillation]] : 거대 모델의 지식을 경량 모델로.

- **Inference Acceleration (속도 최적화)**
    - [[study/Inference Acceleration|🚀 Inference Acceleration Index]] : 추론 가속화 기술 총정리.
    - [[study/KV Cache Optimization|KV Cache & PagedAttention]] : vLLM의 핵심 메모리 기술.
    - [[study/Speculative Decoding|Speculative Decoding]] : 인턴과 부장님 모델의 협업.
    - [[study/Long-seq|Long Context Handling]] : RoPE Scaling과 긴 문맥 처리.
    - [[study/분산 GPU 훈련|Distributed GPU Training]] : 3D Parallelism과 ZeRO 전략.

- **Architecture & Strategies**
    - [[study/MoE|Mixture of Experts (MoE)]] : Sparse 모델의 표준.
    - [[study/RAG|RAG]] & [[study/AI Fallback Strategy|Fallback Strategy]] : 할루시네이션 방지와 안전장치.
    - [[study/Uncertainty_Estimation|Uncertainty Estimation]] : 모델의 불확실성 측정.

---

### Deep Learning Architectures
AI 모델의 뼈대가 되는 핵심 아키텍처들에 대한 심층 분석입니다.

- **Transformer Family**
    - [[study/Transformer|Transformer]] : 모든 것의 시작.
    - [[study/GPT|GPT (Decoder-only)]] : 생성형 모델의 표준.
    - [[study/Attention|Attention]] & [[study/Cross Attention|Cross Attention]] : 메커니즘의 이해.
    - [[study/Encoder & Decoder|Encoder & Decoder]] : 구조적 차이점.

- **Next-Gen & Vision**
    - [[study/Mamba|Mamba (SSM)]] : Transformer를 넘어서는 선형 시간 모델.
    - [[study/ViT|Vision Transformer (ViT)]] : 이미지를 패치로 쪼개는 혁명.
    - [[study/VLM|Vision-Language Model (VLM)]] : 보는 눈을 가진 LLM (CLIP, LLaVA).
    - [[study/Neural Architecture Search|NAS]] : AI가 AI를 설계하는 기술.
    - [[study/Generative Model|Generative Model]] : [[study/GAN|GAN]] & [[study/VAE|VAE]].

---

### Mathematical Foundations
화려한 기술 뒤에 숨겨진 수학적 원리와 통계학적 베이스입니다.

- **Theory of Deep Learning**
    - [[study/NTK|Neural Tangent Kernel (NTK)]] : 무한 너비 신경망의 수학적 증명.
    - [[study/Gaussian Kernel|Gaussian Kernel]] & [[study/Kernel Regression|Kernel Regression]] : NTK의 선수 지식.
    - [[study/SVM (Support Vector Machine)|SVM]] : 마진 최대화의 미학.
    - [[study/Markov Decision Process|MDP]] : 강화학습의 수학적 토대.
    - [[study/GRPO|GRPO]] : DeepSeek-R1의 핵심, 비평가 없는 강화학습.

- **Basic Components**
    - [[study/Loss Functions|Loss Functions]] : MSE부터 Focal Loss까지.
    - [[study/Activation Functions|Activation Functions]] : ReLU, GELU, Swish 비교.
    - [[study/Normalization|Normalization]] : [[study/Layer Normalization|LayerNorm]] & BatchNorm.
    - [[study/Regularization|Regularization]] : Dropout과 Weight Decay.
    - [[study/Weight Initialization|Weight Initialization]] : 초기화의 중요성 (He vs Xavier).

- **Statistics (MATH343 Coursework)**
    - [[study/MATH343 확통연/Chapter 1&2|Ch 1-2. Sample & Statistics]]
    - [[study/MATH343 확통연/Chapter 3. Elements of probability|Ch 3. Probability Axioms]]
    - [[study/MATH343 확통연/Chapter 4. Random variable|Ch 4. Random Variables]]
    - [[study/MATH343 확통연/Chapter 5. Special random variable|Ch 5. Special Distributions]]
    - [[study/MATH343 확통연/Chapter 6. Distributions of sampling statistics|Ch 6. Sampling Distributions]]
    - [[study/MATH343 확통연/Chapter 7. Parameter Estimation|Ch 7. Estimation]]
    - [[study/MATH343 확통연/Chapter 8. Hypothesis Testing|Ch 8. Hypothesis Testing]]

---
<center> <small> Powered by <a href="https://quartz.jzhao.xyz/">Quartz 4</a> © 2025 RicePunchb </small> </center>