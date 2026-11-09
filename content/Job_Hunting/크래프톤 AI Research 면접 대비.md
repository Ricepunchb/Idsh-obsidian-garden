면접 대비 자료를 만들기 위해, 기존에 수집한 논문 정보에 더해 게임별 AI 기능 적용 사례와 내부 AI 활용 현황에 대한 추가 조사를 진행하겠습니다.
충분한 정보를 수집했습니다. 이제 면접 대비 자료를 작성하겠습니다.

---

# 크래프톤 AI Research 인턴십 면접 대비 자료

## Part 1. 논문 기반 연구 방향성 클러스터링

2025~2026년 논문 25편 이상을 분석한 결과, KRAFTON AI의 연구는 **6개 클러스터**로 정리됩니다. 각 클러스터는 "게임이라는 도메인"을 중심으로 유기적으로 연결되어 있습니다.

### 클러스터링 개요

| # | 클러스터 | 핵심 질문 | 대표 논문 |
|---|---|---|---|
| A | **게임 플레이 에이전트 평가·벤치마크** | "AI가 게임을 얼마나 잘 플레이하는가?" | Orak, FlashAdventure, AgentVidBench, A2Z GameSpec-Bench, pubg-winprob |
| B | **확산 모델 (텍스트·이미지 생성)** | "어떻게 생성 품질과 제어력을 높이는가?" | DAS(ICLR'25 Spotlight), R2F(ICLR'25 Spotlight), LookUM(ICML'26), Looped Diffusion LM(NeurIPS'26) |
| C | **LLM 효율화·경량화** | "어떻게 온디바이스에서 돌리는가?" | Lexico(ICML'25), MoE-to-Dense(NeurIPS'26 Oral), Not All Bits Are Equal(ICLR'26) |
| D | **강화학습·정책 최적화 이론** | "어떻게 안정적으로 학습시키는가?" | Beyond Clipping, Is Importance Ratio Necessary(NeurIPS'26), AsyncOPD, WHALE, Chess-R1, Prompt2Policy |
| E | **멀티모달·음성 AI** | "어떻게 보고 듣고 말하는가?" | Raon-Speech/Chat, SitCom(NeurIPS'26), ArtiAgent(CVPR'26), VLM-SubtleBench, Gemini Embeddings |
| F | **AI 안전·정렬 (배포 환경)** | "어떻게 안전하게 서비스하는가?" | ThinkSafe(NeurIPS'26), When Agent Responses Become Interaction State(NeurIPS'26 WS Oral), Aligning Embodied Companion Agent(NeurIPS'26 WS) |

### 방향성을 관통하는 하나의 스토리

이 6개 클러스터를 관통하는 큰 그림은 **"Co-Playable Character(CPC)를 실현하는 전 기술 스택의 내재화"** 입니다.

```
[게임 이해]          [행동 결정]        [소통]           [실행 환경]        [신뢰성]
Orak/AgentVidBench → RL/DPO 이론 →  Raon-Speech →   Lexico/MoE-to-Dense → ThinkSafe
(무엇이 잘하는가)   (어떻게 학습)   (어떻게 대화)    (어디서든 돌아가게)  (안전하게)
                                       ↑
                              [콘텐츠 생성]
                          DAS/R2F/LookUM (월드·에셋)
```

면접에서 "KRAFTON AI의 연구 방향이 어떻다고 생각하나?"라는 질문이 나오면, 이 CPC 스택 관점에서 답하는 것이 가장 강력합니다. 개별 논문을 나열하는 것이 아니라, **"저들은 게임에서 함께 플레이하는 AI를 만들기 위해 평가→학습→생성→경량화→안전이라는 전체 파이프라인을 자체 연구로 해결하고 있다"**는 프레임이 핵심입니다.

---

## Part 2. 클러스터별 논문 상세 분석

### 클러스터 A — 게임 플레이 에이전트 평가·벤치마크

이 클러스터가 KRAFTON AI의 정체성을 가장 잘 보여줍니다. 일반 AI 연구소가 게임을 다루더라도 하나의 게임을 다루지만, KRAFTON은 **"게임 전반에 대한 평가 인프라"**를 구축하고 있습니다.

**① Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games**【turn1search18】【turn3fetch0】
- **구조**: 12개 실제 상용 게임(다양한 장르)에서 LLM 에이전트 성능을 평가하는 foundational 벤치마크
- **핵심 기술**: 전문가 게임플레이 시연(demonstration) 데이터를 활용해, 익숙하지 않은 게임에 모델을 적응(few-shot adaptation)시키는 파이프라인
- **의의**: "오락(오락)"이라는 순우리말에서 이름을 따온 것처럼, 게임 플레이 AI 연구의 표준 기반을 만들겠다는 의도. ICLR 2026 채택 + EMNLP 2025 Wordplay Workshop Outstanding Paper 수상
- **면접 포인트**: "게임사가 아니면 만들 수 없는 벤치마크" — 12개 게임에 대한 라이선스·데이터 접근성은 KRAFTON의 고유 자산

**② FlashAdventure: GUI Agents Solving Full Story Arcs in Diverse Adventure Games**【turn3fetch0】
- **구조**: 34개 Flash 기반 어드벤처 게임에서 전체 스토리 완주(full story arc) 능력 테스트
- **핵심 개념**: **관찰-행동 격차(observation-behavior gap)** — 이전 게임 정보를 기억하고 그에 따라 행동해야 하는 과제. 단편적 액션 수행이 아닌 장기 서사 이해를 요구
- **제안**: CUA-as-a-Judge(자동 게임플레이 평가기), COAST(장기 단서 기억 기반 에이전트 프레임워크)
- **발견**: 현재 GUI 에이전트는 전체 스토리 아크 완주에 큰 어려움을 겪으며, 인간과 최고 에이전트 간 현격한 격차 존재
- **면접 포인트**: CPC가 "친구처럼 함께 플레이"하려면 스토리 맥락 기억이 필수인데, 이를 측정하는 벤치마크

**③ pubg-winprob: Real-Time Win Probability Prediction in Battle Royale Games via Survival Analysis**【turn9fetch0】
- **구조**: 배틀로얄 장르 특성(100명 동시 시작, 점진적 감원)을 생존 분석(survival analysis) 프레임으로 모델링해 실시간 승률 예측
- **실서비스 연결**: 이 연구는 실제 PUBG e스포츠 중계에 적용되어 실시간 승률 표시, 이동 경로·교전 예측, AI 하이라이트 생성에 활용됨【turn0search1】
- **면접 포인트**: "연구→실서비스 배포" 사이클이 완결된 사례. 면접에서 "연구 성과의 게임 적용"을 논할 때 최고의 예시

**④ AgentVidBench: A Multi-Hop Video Question Answering Benchmark for Evaluating MLLM Agents**【turn9fetch0】
- **구조**: 멀티홉(multi-hop) 추론이 필요한 비디오 QA로 MLLM 에이전트 평가
- **의의**: 게임 영상을 보고 "저 적은 어디서 왔고, 아까 그 아이템을 썼는가"처럼 시간적으로 떨어진 정보를 연결해 답해야 하는 능력 측정

**⑤ A2Z GameSpec-Bench: How Faithfully Can Coding Agents Generate Games from Game Design Specifications?**【turn9fetch0】
- **구조**: 게임 기획서(스펙)를 입력하면 코딩 에이전트가 게임을 생성하는 충실도 평가
- **의의**: AI로 게임을 만드는 시대의 "품질 측정". 2026년 10월 가장 최신에 공개

### 클러스터 B — 확산 모델 (텍스트·이미지 생성)

**⑥ Test-time Alignment of Diffusion Models without Reward Over-optimization (DAS)** — ICLR 2025 **Spotlight**【turn1search10】【turn11fetch0】
- **문제**: 확산 모델을 특정 보상(예: 미적 품질)에 정렬하면 reward hacking(over-optimization) 발생, 다양성 상실
- **해법**: 학습 없이 test-time에 Sequential Monte Carlo(SMC) 샘플링으로 보상 정렬된 분포에서 직접 샘플링. tempering 기법으로 안정화
- **성과**: fine-tuning과 대등하거나 우수한 보상 달성 + 다양성·cross-reward 일반화 보존
- **게임 적용**: 게임 에셋을 특정 스타일로 생성하면서 다른 스타일 능력은 유지

**⑦ Rare-to-Frequent (R2F)** — ICLR 2025 **Spotlight**【turn11fetch0】
- **문제**: 확산 모델은 학습 데이터에서 드물게 나타나는 콘셉트 조합(예: "보라색 코끼리가 우산을 쓴 모습")을 잘 생성 못함
- **해법**: LLM 가이던스로 rare concept을 frequent concept의 조합으로 재해석해 생성
- **게임 적용**: 게임 세계관의 희귀 조합 아이템/캐릭터/씬 생성

**⑧ Lookahead Unmasking (LookUM)** — ICML 2026【turn2fetch0】
- **문제**: Masked Diffusion 언어 모델(MDM)의 성능이 unmasking 순서에 결정적으로 의존하는데, 기존 confidence 기반 샘플링은 근시안적(myopic)
- **해법**: 샘플링을 "가능한 모든 unmasking 순서에 대한 경로 선택"으로 재정식화. 경로 생성기 + 불확실성 기반 검증기 구조로 importance sampling
- **핵심 발견**: 오류가 있는 unmasking이 sequence-level 불확실성을 측정 가능하게 부풀린다는 점을 활용해 오류 경로 회피
- **놀라운 결과**: base LLaDA + LookUM이 RL로 post-training된 LLaDA 1.5와 대등. 단 2-3개 경로만으로 peak 달성
- **면접 포인트**: "RL 없이 디코딩 전략만으로 RL 수준 성능" — 확산 LM 실용화의 중요한 이정표

**⑨ Looped Diffusion Language Models** — NeurIPS 2026【turn2fetch0】【turn1fetch2】
- **핵심**: 소수 transformer 레이어를 반복 재사용(loop)해, 파라미터 추가 없이 훨씬 적은 학습 컴퓨팅으로 baseline 품질 매칭
- **의의**: 모델 크기(파라미터) 대신 깊이(반복)를 조절하는 새로운 스케일링 축 제시. 경량 온디바이스 언어모델에 직결

### 클러스터 C — LLM 효율화·경량화

이 클러스터는 **"inZOI의 Smart Zoi가 온디바이스에서 돌아간다"**는 실제 제약 조건과 직결됩니다.

**⑩ Lexico: Extreme KV Cache Compression via Sparse Coding over Universal Dictionaries** — ICML 2025【turn1search1】【turn1search10】
- **구조**: KV 캐시를 universal dictionary 기반 sparse coding으로 극단적 압축
- **의의**: LLM 추론 메모리 대폭 절감 → 온디바이스 실행 가능성 확대

**⑪ Pruning and Distilling Mixture-of-Experts into Dense Language Models** — NeurIPS 2026 **Oral**【turn1fetch0】
- **구조**: 거대 MoE 모델(전문가 여러 개)을 작은 dense 모델로 프루닝+증류
- **의의**: 서버급 MoE 모델의 지식을 온디바이스급 dense 모델로 이전. NeurIPS Oral 선정은 이 분야 최상위 인정
- **연결**: A.X K2 Raon-Speech가 21B-A3B(MoE)인데, 이를 게임에 넣으려면 dense 모델로의 변환 기술이 필요

**⑫ Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models** — ICLR 2026【turn1search7】【turn1search9】
- **핵심 발견**: 추론 모델의 메모리 최적화는 비추론 모델과 근본적으로 다름. 1,700+ 시나리오 분석 결과, **스케일(모델 크기·토큰 예산)에 따라 최적 양자화 비트 수가 달라진다**는 것
- **제안**: R-KV라는 KV cache eviction 방법
- **면접 포인트**: "4-bit quantization이 만능이 아니다"라는 통설 뒤집기. 오버클라우드 의존 없는 온디바이스 추론의 실무 지침

### 클러스터 D — 강화학습·정책 최적화 이론

이 클러스터는 Core Research 팀(이강욱 CAIO 직속)의 이론적 깊이를 보여주는 영역입니다.

**⑬ Beyond Clipping: Signed Logarithmic Smoothing for Policy Optimization** — NeurIPS 2026【turn1fetch0】
- PPO의 clipping 메커니즘을 대체하는 새로운 정책 최적화 방법. 정책 업데이트의 안정성 이론에 기여

**⑭ Is the Importance Ratio Necessary for Stable Reinforcement Learning in LLMs?** — NeurIPS 2026【turn1fetch0】
- LLM RL에서 importance ratio의 필요성을 근본적으로 재검토. GRPO/PPO 계열 알고리즘의 설계 선택을 재고하게 만드는 연구

**⑮ AsyncOPD: How Stale Can On-Policy Distillation Be?** — NeurIPS 2026【turn1fetch0】
- On-policy distillation에서 데이터 최신성(staleness) 허용 범위 분석. 비동기 학습 파이프라인 설계의 이론적 기반 제공

**⑯ Chess-R1: Can LLMs Develop Strategic Reasoning?**【turn3fetch0】
- **핵심 발견**: chess-pretrained 가치 네트워크의 dense reward(지식 증류)로 LLM RL 학습 → sparse binary reward보다 우수하지만, **모든 모델이 전문가 수준에 크게 못 미치는 plateau에 도달**
- **원인 분석**: 사전학습 모델의 내부 체스 이해 자체가 결핍되어 있어 RL만으로는 극복 불가
- **면접 포인트**: "게임 전략 추론의 근본 한계"를 정직하게 규명한 연구. CPC가 게임을 잘하려면 사전학습 단계부터 게임 이해가 필요하다는 시사점

**⑰ WHALE: Weight-Harness Alternating Learning**【turn9fetch0】
- LLM 에이전트의 agentic RL을 위한 가중치-하네스 교차 학습. 에이전트 프레임워크(harness)와 모델 가중치(weight)를 번갈아 최적화

**⑱ Prompt2Policy**【turn9fetch0】
- "Agentic Engineering for Reinforcement Learning" — 프롬프트로 강화학습 정책을 엔지니어링하는 접근

### 클러스터 E — 멀티모달·음성 AI

**⑲ Raon-Speech / Raon-SpeechChat** (기술 보고서 상세)【turn0search16】【turn0search3】【turn0search18】
- **구조**: Qwen3 백본(36 layers) + Qwen3OmniMoeAudioEncoder(24 layers) + Mimi codec(32 quantizers) + ECAPA-TDNN 화자 인코더
- **3단계 학습**: (1) 음성 모듈 정렬 → (2) 지식 증류 기반 end-to-end SpeechLM 사전학습 → (3) multi-reward DPO post-training
- **데이터**: 138만 시간의 정제된 영어/한국어 음성-텍스트 데이터
- **Raon-SpeechChat의 3가지 혁신**: (1) 스트리밍 입력을 위한 causal encoder, (2) 사용자 음성-어시스턴트 텍스트-어시스턴트 음성의 token-level interleaving, (3) full-duplex(동시 듣기+말하기) 구현
- **성능**: 백채널링("응", "그래" 같은 반응), 중단 처리, 턴테이킹, 겹침 대화에서 최고 수준. 단일 GPU에서 실시간보다 빠른 스트리밍, 서브초 time-to-first-token
- **면접 포인트**: STT→LLM→TTS 직렬 구조가 아닌 **음성을 일급 모달리티로 직접 이해·생성**하는 구조. 게임 내 음성 대화 CPC의 핵심 기술

**⑳ A.X K2 Raon-Speech (21B-A3B)** — 2026년 7월【turn4fetch2】
- SKT A.X K2 Light(20B-A3B MoE) + KRAFTON 자체 학습 음성 인코더/코덱
- **한국어 종합 1위, 영어 3위** (공개 30B 이하 SpeechLM 중)
- 국내 최초 독자 개발 음성 파운데이션 모델

**㉑ SitCom: Scaling Egocentric Multi-Party Spoken Dialogue** — NeurIPS 2026【turn1fetch0】
- 1인칭 시점 다자간 음성 대화 데이터셋 스케일링. 상황 인식 커뮤니케이션 지원
- CPC가 여러 명과 동시에 대화하는 상황(파티 플레이)에 필요한 기술

**㉒ ArtiAgent** — CVPR 2026【turn3search10】【turn10fetch0】
- VLM과 확산 모델이 시각적 아티팩트(결함)를 인지하고 수정하도록 agentic data synthesis로 학습
- 게임 에셋 AI 생성 시 자동 품질 검수·수정

**㉓ VLM-SubtleBench**【turn3fetch0】
- 시각적으로 유사한 이미지 간 미묘한 차이 인식 능력 평가. 10가지 차이 유형(속성/상태/감정/시간/공간 등) + 산업·의료·항공 도메인
- VLM과 인간 간 체계적 격차 규명

**㉔ Can Gemini Embeddings Be a Multimodal Encoder for LLMs?**【turn1search16】【turn1search17】
- Gemini embedding API를 frozen LLM(Qwen3-4B)의 멀티모달 인코더로 사용. 17M MLP adaptor만으로 객체 분류 97%, 명령 전사 89%
- 커스텀 인코더 학습 없이 멀티모달 능력 추가하는 "저비용 레시피" 탐구

### 클러스터 F — AI 안전·정렬 (배포 환경)

이 클러스터는 **"실제 서비스에 AI를 넣었을 때 생기는 문제"**를 다룬다는 점에서 특히 주목할 만합니다. 논문 제목 자체가 실서비스 경험을 반영합니다.

**㉕ ThinkSafe: Self-Generated Safety Alignment for Reasoning Models** — NeurIPS 2026【turn1fetch2】【turn3fetch0】
- 추론 모델이 스스로 안전 학습 데이터(safety-training traces)를 생성 → 유해 프롬프트 저항성 회복 + 원래 문제 해결 능력 보존

**㉖ When Agent Responses Become Interaction State: Safety in a Deployed AI Teammate** — NeurIPS 2026 Workshop **Oral**【turn1fetch0】
- **제목 그대로**: 실제 배포된 AI 팀원(PUBG Ally)의 응답이 게임의 상호작용 상태가 될 때 발생하는 안전 문제 분석
- Workshop Oral 선정 — 게임 배포 환경의 안전성 연구로 국제적 인정

**㉗ Aligning Embodied Companion Agent Evaluation with User Experience** — NeurIPS 2026 Workshop【turn1fetch0】
- 게임 내 embodied companion agent(CPC) 평가를 사용자 경험과 정렬
- "벤치마크 점수가 높다"와 "유저가 만족한다"의 간극을 메우는 방법론

---

## Part 3. 게임별 AI 기능 적용 현황 (논문 외 최신 소식)

### PUBG: BATTLEGROUNDS — 가장 성숙한 AI 적용 사례

**PUBG Ally (세계 최초 CPC)**【turn0search5】【turn0search9】【turn0search1】
- **개념**: 솔로 플레이어와 실시간 음성으로 대화하며 함께 플레이하는 AI 팀원. NVIDIA ACE 기반
- **타임라인**: CES 2025 최초 공개 → 2025년 10월 30일 플레이테스트 계획 발표 → 2026년 초 PUBG Arcade 테스트 → **2026년 6월 Update 42.1에서 "Ally Duo" 한정 베타 모드로 정식 배포** (PC 전용)
- **기술 성과**: 게임 화면과 음성 명령을 동시 이해, 평균 응답 속도 **0.8초**까지 단축
- **설계 철학**: 성준식 실장 — "AI 봇이 너무 잘하면 이용자가 싫어해... **사람 같은 실수를 하는** 방향으로 설계"【turn0search3】 이것은 ThinkSafe·Agent 안전 논문 클러스터와 직결되는 실무 고민

**e스포츠 중계 AI**【turn0search1】【turn0search14】
- 프로 경기 데이터 학습 모델로 실시간 승률 예측, 이동 경로·교전 예측, AI 하이라이트 자동 생성
- 캐스터가 경기를 더 깊이 있게 설명하도록 데이터 지원
- pubg-winprob 논문이 실제 여기에 적용됨

**안티치트 AI**【turn0search10】【turn0search1】
- 플레이 패턴 분석 기반 AI 탐지 시스템을 1년 이상 운영, **하루 1만 건 이상** 핵 의심 사례 탐지
- ESP(벽 너머 적 위치 확인) 등 기존 방식으로 잡기 어려운 핵을 AI 데이터 분석으로 탐지
- 2026년 콘솔 제재 계정 수 전년 대비 44.6% 증가

**RL Bot**【turn1fetch2】
- 강화학습 기반 AI 봇으로 사람과 유사한 전략 플레이. 유저 몰입감 + 개발자의 시나리오 수동 설계 부담 경감

### inZOI — 라이브 게임 최초 CPC 탑재

**Smart Zoi**【turn1fetch2】【turn1search1】
- 라이브 게임(얼리 액세스) 최초로 CPC 기술 도입
- 페르소나 기반: Zoi Pen으로 플레이어가 성격을 설정하면 Zoi가 스스로 상황 판단·반응
- **SLM(소형 언어모델)로 온디바이스에서 안정적 실행** — Lexico, MoE-to-Dense 등 경량화 연구 클러스터가 지원하는 제품
- 영어 + 한국어 지원 확장 중【turn2find2】

**3D 프린터 (AI 에셋 생성)**【turn1search1】【turn0search0】
- 플레이어가 업로드한 2D 이미지를 AI가 3D 에셋(가방, 신발, 소품 등)으로 변환해 게임 내 배치
- R2F, DAS 같은 확산 모델 연구의 실제 게임 적용

### OVERDARE (오버데어) — AI 에이전트로 게임 제작 자체를 바꾸는 UGC 플랫폼

**스튜디오 에이전트 (Studio Agent)**【turn0search16】 (2026년 4월 8일 공개)
- 크리에이터가 채팅창에 **자연어로 지시**하면 AI가 게임 에디터를 직접 제어해 게임을 구현
- 스크립트를 포함한 오브젝트 생성·편집·적용을 자율 수행, 결과물을 에디터에서 실시간 확인 후 추가 지시 가능
- AI와 수동 편집을 병행하는 유연한 제작 방식
- 커스텀 언리얼 엔진 5 기반, AI가 Lua 스크립트를 자동 생성해 적용
- **확장 로드맵**: 아트 에셋 배치, 레벨 디자인까지 AI가 수행 → 게임 전반을 AI로 제작하는 환경 구축 목표

**AI 코스튬 (AI Costume)**【turn0search16】
- 아바타 의상 텍스처를 자동 생성. 원하는 스타일 입력만으로 다양한 의상 빠르게 생성

### 렐루게임즈 (ReLU Games) — "딥러닝이 융합된 게임 플레이" 전문 스튜디오

- **미션**: "게임의 본질적인 재미를 발견하는 도구로 딥러닝을 탐구"【turn0search5】
- **언커버 더 스모킹 건**: 선택지 고르는 방식이 아닌, **챗GPT 기반 캐릭터와의 대화·설득 과정**을 거쳐야 하는 추리 게임【turn0search9】
- **푼다: AI 퍼즐** 등: 크래프톤 개발 음성 인식 AI 탑재, 그래픽·음성 에셋 대부분 생성형 AI로 제작【turn0search8】
- **포지셔닝**: 다른 스튜디오가 "게임에 AI를 넣는다"면 렐루는 "AI 없이는 존재할 수 없는 게임 장르"를 개척【turn0search8】

### 그 외 조직 단위 AI 활용

- **루도 로보틱스 (Ludo Robotics)**: 게임 시뮬레이션 역량을 피지컬 AI·로보틱스로 확장. 미국 모회사 + 한국 법인(2026년 2월 설립)【turn2find0】
- **서울대 월드모델 공동연구**: 게임 가상환경과 로봇 현실환경 공통 예측·판단 기술. 3년간 공동연구센터 운영 (2026년 10월 출범)【turn2find0】

---

## Part 4. 사내 AI 활용 현황 ("AI First"의 실체)

### 조직 문화로서의 AI

- **AI First 전환 선언** (2025년 10월): Agentic AI 중심 업무 자동화, 1,000억 원 이상 투자, 2026년부터 매년 300억 원 AI 도구 지원 (기존 10배)【turn2find0】
- **직원 AI 활용률 97.6%** (2026년 2월 전사 서베이 기준)【turn0search0】【turn0search2】 — NDC 26(2026년 6월)에서 공식 발표된 수치
- NDC 26 세션 "AI 전환, 효율화 아닌 업무 본질의 변화"에서 넥슨(바텀업)과 크래프톤(탑다운)의 AX 접근법 비교 발표【turn0search0】【turn0search1】
- 사내 AI 러닝 허브, AI 라운드테이블, AI 해커톤 운영【turn2find0】
- 채용 전형에 **AI 역량평가** 단계가 별도로 존재 (2027 인턴십부터)【turn6fetch0】

### 게임 제작 파이프라인의 AI

- 크래프톤의 AI 활용은 "제작비 절감보다 **게임의 규칙과 이용자 경험을 바꾸는 데** 무게"【turn0search17】
- 라이브 서비스에서 축적한 AI 경험을 다른 스튜디오로 확대 적용하는 것이 전략 (성준식 실장)【turn0search1】
- 위에서 아래로(Top-down) AI 퍼스트를 밀어붙이는 방식이 넥슨의 바텀업 방식과 대비됨【turn0search1】

---

## Part 5. 면접 대비 핵심 정리

### 5분 요약 스크립트 (순서대로 답변 구성)

> "KRAFTON AI의 연구는 크게 6개 방향으로 클러스터링됩니다: ① 게임 플레이 에이전트 평가(Orak 등), ② 확산 모델 생성 품질(LookUM 등), ③ LLM 경량화(Lexico 등), ④ 강화학습 이론, ⑤ 멀티모달·음성(Raon 시리즈), ⑥ 배포 환경 AI 안전성입니다. 이 모든 것이 하나로 수렴하는 지점이 **CPC(Co-Playable Character)**입니다. 게임을 이해하고, 학습하고, 음성으로 대화하며, 온디바이스에서 돌아가고, 안전하게 서비스되는 AI 친구를 만드는 전 스택을 자체 연구로 해결하고 있죠.
>
> 특히 주목할 점은 연구가 실제 서비스와 맞닿아 있다는 것입니다. pubg-winprob 논문은 실제 PUBG e스포츠 중계의 실시간 승률 예측에 쓰이고, Raon-Speech는 PUBG Ally의 음성 대화 기반이며, Lexico·MoE-to-Dense 같은 경량화 연구는 inZOI Smart Zoi가 온디바이스에서 돌아가게 하는 기술입니다. 'When Agent Responses Become Interaction State'처럼 배포된 AI 팀원의 안전성을 다루는 논문 제목 자체가 실서비스 경험을 반영하고 있습니다."

### 예상 질문과 답변 방향

**Q1. "KRAFTON AI 연구 중 가장 흥미로웠던 논문이 무엇인가?"**
- **전략**: 자신의 전공/관심 분야와 연결되는 논문 1편을 깊게 준비
- 추천: LookUM (RL 없이 디코딩만으로 RL 성능 — 확산 LM 실용화), Chess-R1 (전략 추론의 근본 한계 — 솔직한 negative result의 가치), Orak (게임사만 만들 수 있는 벤치마크)

**Q2. "게임 회사에서 AI를 연구하는 것의 의미가 무엇이라고 생각하나?"**
- **핵심 답변 요소**: ① 대규모 실시간 상호작용 데이터 (수백만 플레이어의 행동·음성·전략), ② 즉각적인 검증 환경 (실제 게임 배포), ③ 시뮬레이션·가상세계 구축 역량 (월드모델→로봇 확장), ④ 게임이라는 복잡한 멀티모달 도메인 (규칙+전략+감정+소통)

**Q3. "왜 다른 AI 기업이 아니라 크래프톤인가?"**
- **핵심**: 일반 LLM 기업은 벤치마크 성능이 목표지만, KRAFTON은 **"유저가 친구처럼 느끼는 AI"**가 목표. 이는 성능 지표가 아닌 경험 설계 문제이며, 게임 도메인에서만 풀 수 있는 문제. "AI 봇이 너무 잘하면 싫어한다"는 성준식 실장의 발언이 이를 상징

**Q4. "CPC 기술의 가장 큰 기술적 난제는 무엇이라고 보나?"**
- 후보 답변: (a) 온디바이스 SLM의 품질-지연 트레이드오프 (0.8초 응답 달성), (b) 장기 맥락 기억 (FlashAdventure가 보여준 관찰-행동 격차), (c) 안전성 (ThinkSafe, 배포형 AI 안전), (d) 진짜 사람 같은 실수 만들기

**Q5. "Raon-Speech를 아는가?"**
- **핵심 구조 답변**: STT→LLM→TTS 직렬이 아닌 **음성을 일급 모달리티로 직접 이해·생성**하는 SpeechLM. Qwen3 백본 + 음성 인코더/코덱을 3단계(정렬→증류 사전학습→DPO post-training)로 결합. Raon-SpeechChat은 causal encoder + token-level interleaving으로 full-duplex(동시 듣기/말하기) 구현. A.X K2 Raon-Speech(21B)는 한국어 1위

### 반드시 기억할 숫자·날짜

| 항목 | 수치 |
|---|---|
| 3대 학회 누적 논문 | 85편 |
| ICML 2026 | 20편 (메인 10편, 한국 게임사 1위·글로벌 3위) |
| 2025년 3대 학회 | 15편 (ICLR 6편 중 3편 Spotlight) |
| Raon-Speech 학습 데이터 | 138만 시간 |
| PUBG Ally 응답 속도 | 0.8초 |
| 안티치트 일일 탐지 | 1만 건 이상 |
| 직원 AI 활용률 | 97.6% (2026.2) |
| AI First 투자 | 1,000억 원+, 연 300억 AI 도구 지원 |
| 딥러닝본부 설립 | 2022년 (2021년 AI 투자 시작) |
| 이강욱 CAIO 선임 | 2026년 2월 |

### 최신 뉴스 3가지 (면접 직전 체크)

1. **2026년 6월**: PUBG Update 42.1로 "Ally Duo" 베타 배포 개시【turn0search9】
2. **2026년 10월 8일**: 서울대와 월드모델 공동연구 출범 (3년간, 게임+로봇)【turn2find0】
3. **2026년 6월 NDC 26**: 직원 97.6% AI 활용 공개, "효율화 넘어 업무 본질 변화" 발표【turn0search0】

---

이 자료의 모든 정보는 공개 자료(krafton.ai, 크래프톤 보도자료, arXiv, GitHub, 언론 보도) 기반이며, 면접 전 각 논문의 원문 초록(arXiv 링크)을 최소 2-3편 직접 읽어보는 것을 강력히 권합니다. 특히 자신이 지원한 세부 직무(LLM/RL/멀티모달/Speech)에 해당하는 클러스터의 논문은 아브스트랙 수준이 아니라 방법론까지 이해하고 가는 것이 좋습니다.