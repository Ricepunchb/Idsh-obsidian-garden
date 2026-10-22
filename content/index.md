---
title: Suhyun Bae · 배수현
description: AI Research Engineer
publish: True

# tags:
#   - example-tag
---

> [!profile] AI Research Engineer
> **Mathematical AI Lab**, Korea University · Seoul, Korea
>
> *Math for AI. Make efficient and robust AI with mathematics.*
>
> [Email](mailto:baeshstar@gmail.com) · [GitHub](https://github.com/Ricepunchb) · [LinkedIn](https://www.linkedin.com/in/suhyun-bae-869043286) · [CV](Job_Hunting/Portfolio.htm)

## About

안녕하세요, **배수현(Suhyun Bae)** 입니다. 고려대학교 산학협력단 **수리인공지능 연구실**에서 연구원으로 일하고 있으며, 같은 연구실에서 이동헌 교수님의 지도를 받는 행운을 누리며 수학과 석사를 마쳤습니다.

제 연구는 수학적 도메인 지식과 수학에서 가져온 아이디어를 바탕으로, **모달리티(텍스트·이미지·수치 등)와 모델 구조(Transformer 여부)에 구애받지 않고** 아키텍처를 최적화하고 효율적으로 만들며, 나아가 **근본적으로 신뢰할 수 있게(reliable)** 만드는 것을 목표로 합니다. 예를 들어 Adele Ring, 조화해석학 같은 대수적 구조를 이용해 언어모델의 수치 임베딩과 신뢰성(hallucination)을 개선하고 있습니다. 데이터에서 구조를 "학습"하기 전에, 수학이 이미 알려주는 구조를 먼저 활용할 수 있다고 믿습니다.

> [!interests] Research Interests
> - **Representation Learning**
> - **Efficient Deep Learning**
> - **Autonomous Agents**

## News

> [!news]+ Recent Updates
> - `2026.07` **Prime Fourier Embeddings** — *ICML 2026 Workshop AI4MATH* 발표
> - `2026.04` 고려대학교 산학협력단 기초과학연구원 지속가능자원-환경연구센터 연구원으로 합류
> - `2026.02` 고려대학교 수학과 석사 졸업
> - `2025.12` **Numbers Already Carry Their Own Embeddings** — *NeurIPS 2025 Workshop MATH-AI*
> - `2025.10` **OPC: One-Point-Contraction Unlearning** — *ICCV 2025 Workshop U&ME*

## Publications

<small>† equal contribution</small>

> [!paper] ICML 2026 Workshop · AI4MATH
> **[Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic](https://arxiv.org/abs/2606.23044)**  
> Hyunsang Hwang†, **Suhyun Bae**†, Donghun Lee
>
> 조화해석학의 **Pontryagin 쌍대성**에 기반해 정수를 소수-인덱스 (cos, sin) 쌍으로 인코딩하는 임베딩을 제안하고, equivariant한 선형사상이 소수별 **block-diagonal 구조**를 가져야 함을 증명.
>
> [arXiv](https://arxiv.org/abs/2606.23044) · `2026.07`
>
> > [!cite]- BibTeX
> > ```bibtex
> > @inproceedings{hwang2026prime,
> >   title     = {Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic},
> >   author    = {Hwang, Hyunsang and Bae, Suhyun and Lee, Donghun},
> >   booktitle = {ICML 2026 Workshop on AI for Mathematics (AI4MATH)},
> >   year      = {2026},
> >   eprint    = {2606.23044},
> >   archivePrefix = {arXiv},
> >   note      = {Equal contribution}
> > }
> > ```

> [!paper] NeurIPS 2025 Workshop · MATH-AI
> **[Numbers Already Carry Their Own Embeddings](https://arxiv.org/abs/2606.14108)**  
> **Suhyun Bae**, Donghun Lee
>
> 대수정수론의 **Adele Ring**을 학습 없이(training-free) 결정론적 수치 임베딩으로 사용해, 숫자의 크기·패턴을 보존하는 임베딩 공간을 제안.
>
> [arXiv](https://arxiv.org/abs/2606.14108) · `2025.12`
>
> > [!cite]- BibTeX
> > ```bibtex
> > @inproceedings{bae2025numbers,
> >   title     = {Numbers Already Carry Their Own Embeddings},
> >   author    = {Bae, Suhyun and Lee, Donghun},
> >   booktitle = {NeurIPS 2025 Workshop on Mathematical Reasoning and AI (MATH-AI)},
> >   year      = {2025},
> >   eprint    = {2606.14108},
> >   archivePrefix = {arXiv}
> > }
> > ```

> [!paper] ICCV 2025 Workshop · U&ME
> **[OPC: One-Point-Contraction Unlearning Toward Deep Feature Forgetting](https://arxiv.org/abs/2507.07754)**  
> Jaeheun Jung, Bosung Jung, **Suhyun Bae**, Donghun Lee
>
> Machine Unlearning이 출력값만 바꾸고 내부 표현에는 정보가 잔존하는 **"표면적 망각(superficial forgetting)"** 현상을 지적하고, 은닉층 특징 복원을 통해 실제 망각 여부를 검증하는 평가 기법을 제안.
>
> [arXiv](https://arxiv.org/abs/2507.07754) · `2025.10`
>
> > [!cite]- BibTeX
> > ```bibtex
> > @inproceedings{jung2025opc,
> >   title     = {OPC: One-Point-Contraction Unlearning Toward Deep Feature Forgetting},
> >   author    = {Jung, Jaeheun and Jung, Bosung and Bae, Suhyun and Lee, Donghun},
> >   booktitle = {ICCV 2025 Workshop on Unlearning and Model Editing (U\&ME)},
> >   year      = {2025},
> >   eprint    = {2507.07754},
> >   archivePrefix = {arXiv}
> > }
> > ```

> [!paper] KCC 2024 · 한국정보과학회
> **[Exploring Hallucination Types in Question-Answering Generation and Limitation of Text Evaluation Metrics](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11861875)**  
> **Suhyun Bae**, Donghun Lee
>
> LLM이 QA 생성 시 유발하는 텍스트 환각을 **5가지 유형**으로 분류하고, BLEU/METEOR/ROUGE-L 등 기존 텍스트 유사도 지표가 유형별로 탐지력이 다름을 실험으로 규명.
>
> [DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11861875) · `2024.06` · pp. 450–452
>
> > [!cite]- BibTeX
> > ```bibtex
> > @inproceedings{bae2024exploring,
> >   title     = {Exploring Hallucination Types in Question-Answering Generation and Limitation of Text Evaluation Metrics},
> >   author    = {Bae, Suhyun and Lee, Donghun},
> >   booktitle = {Proceedings of the Korea Computer Congress (KCC)},
> >   year      = {2024},
> >   pages     = {450--452}
> > }
> > ```

## Experience

> [!experience] Work
> - `2026.04 – Present` **Researcher** · 고려대학교 산학협력단 지속가능자원-환경연구센터
> - `2024.07 – 2025.02` **Data Scientist Intern** · SK매직

## Education

> [!education] Degrees
> - `2024.03 – 2026.02` **M.S. in Mathematics** · 고려대학교 (Advisor: Prof. Donghun Lee)
> - `2018.03 – 2024.02` **B.S. in Mathematics** · 고려대학교

## Contact

> [!contact] Get in touch
> - **Email** — [baeshstar@gmail.com](mailto:baeshstar@gmail.com)
> - **GitHub** — [@Ricepunchb](https://github.com/Ricepunchb)
> - **LinkedIn** — [suhyun-bae](https://www.linkedin.com/in/suhyun-bae-869043286)
> - **Resume** — [CV](Job_Hunting/Portfolio.htm)

---
<center><small>Powered by <a href="https://quartz.jzhao.xyz/">Quartz 4</a> · © 2025–2026 RicePunchb</small></center>
