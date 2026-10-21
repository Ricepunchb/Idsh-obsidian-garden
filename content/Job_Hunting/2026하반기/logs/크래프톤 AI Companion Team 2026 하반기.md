---
publish: false
title: 크래프톤 AI Companion Team 2026 하반기
회사명: 크래프톤
지원직무: "[AI Research Div.] AI Companion Team (VLA Game Agent)"
상태: 서류제출
서류마감: 상시채용. 2026.10.20 제출
서류발표:
---
> **연결된 제출 서류**:
> - **커버레터**: [[크래프톤 2026 하반기 커버레터]]
> - **국문 CV**: `content/Job_Hunting/KOR_CV.tex`
> - **포트폴리오**: `content/Job_Hunting/2026하반기/자소서들/크래프톤_포트폴리오.html`
> *(※ 전사 대규모 AI Research 통합 공채 공고 및 자기소개서는 [[크래프톤 2026 하반기]] 및 [[크래프톤 2026 하반기 자소서]]로 분리 관리)*

---

# 1. 팀 개요 및 채용 부스 인터뷰 1st-Party 페르소나

- **소속 본부**: 크래프톤 AI Research 본부 (HQ)
- **전담 팀명**: **AI Companion Team** (PUBG AI 팀원 `PUBG Ally`, inZOI CPC `Smart Zoi` 개발 주도)

## [부스 인터뷰 핵심 인사이트: AI Companion Team 페르소나]
1. **조직 내 위상 (Research Head)**:
   - 게임 엔진 실장/서비스 개발 전문팀과 분리된 **상위 코어 AI Research 팀**.
   - PUBG AI 팀원(PUBG Ally)과 inZOI CPC(Smart Zoi)의 원천 R&D 산실.
2. **연구 철학 (Question-Driven & Modality-Agnostic)**:
   - 특정 모달리티(LLM/Vision/Speech)에 갇히지 않고, **"유저에게 어떤 상호작용/플레이 경험을 줄 것인가"**라는 최종단 Research Question에서 출발.
   - 문제 정의에 맞춰 화면, 텍스트, 게임 상태를 행동으로 변환하는 **VLA(Vision-Language-Action) 전용 모델을 처음부터 개발**.
3. **핵심 엔지니어링 과제 (Real-time & Resource Constraints)**:
   - 인게임 실시간 환경에서의 지연(Latency)과 자원 한계를 극복하는 **Gaming SLM/VLA 모델링·추론 최적화**.
   - 파라미터 증량 없이 소형 모델의 추론 및 지시 이행(Instruction-following) 능력을 끌어올리는 아키텍처.
4. **인재상 & 핏**:
   - 수학적 원천 역량을 바탕으로 제약 속 아키텍처 설계와 최적화를 자유롭고 빠르게 돌파할 수 있는 연구자.
   - 단일 방법론에 매몰되지 않고 가설 수립 $\to$ 실패 케이스 분석 $\to$ 모델 아키텍처 개선 피봇이 빠른 사람.
5. **명시적 배제 영역**:
   - **월드모델(World Model)**은 연구 축이 다름 (서울대 공동연구센터 및 별도 축).

---

# 2. 공식 채용공고 원문 (AI Companion Team 개별 공고 2건)

## 공고 ① [주력 타깃] `[AI Research Div.] Research Engineer - Vision Language Action Game Agent (2년 이상 / 인턴)`
- **채용 링크**: [KRAFTON 채용공고 (job=4648)](https://www.krafton.com/careers/recruit-detail/?var_page=1&job=4648&search_list_cnt=10&search_keyword=ai%20research%20div) (`Greenhouse ID: 8417764002`)
- **근무지**: 역삼 센터필드 West 타워
- **고용형태**: 인턴 (6개월) / 상시채용

### [공고문 원문]
> **우리 팀(프로젝트)을 소개합니다.**
>
> **[AI Research 본부 비전]**
> 크래프톤 AI Research 본부는 사내외 여러 분야와 협업하여 다양한 문제에 대한 AI 솔루션을 제공하며, 자체 딥러닝 기술 연구를 통해 우리만의 서비스를 개발합니다. 그 방향성은 크게 두 가지입니다.
> 1. 게임 제작에 도움이 될 수 있는 딥러닝 기술 개발을 통한 게임 제작 효율성 및 보편성 증대.
> 2. 고유의 딥러닝 기술을 개발하여 이를 게임 내/외적으로 다양한 어플리케이션으로 적용.
>
> **[R&D]**
> AI Companion Team은 KRAFTON 내부 게임에서 LLM, SLM, VLM을 적용해 유저와 상호작용하고, 전략 수립 및 협동이 가능한 AI Companion을 개발합니다.
> - 실시간 게임 환경에서 AI 에이전트가 플레이어와 한 팀이 되어 함께 호흡하며 몰입도 높은 플레이 경험을 제공하는 것을 목표로 합니다.
> - 게임 상황을 스스로 파악하고, 유저에게 선제적인 가이드를 제공함으로써 안정적인 적응을 돕는 것을 목표로 합니다.
> - 최적화된 모델 설계부터 운영까지 전 과정을 주도적으로 수행합니다.
>
> **[Culture Fit]**
> AI Research 본부 인원들은 본인 분야에만 묶여있지 않고 서로 다른 프로젝트 팀원들과 교류 및 협업하여 다양한 문제에 창의적인 아이디어를 제시할 수 있습니다. 다양한 문화적 배경을 가진 인원들이 모여 팀을 이루고 있으며 활발한 소통을 위해 언어적 장벽을 해소해줄 수 있는 방법들을 적극 지원하고 있습니다.
>
> **[우리 팀과 함께할 미션]**
> - 게임 환경에서 멀티모달 입력(화면/텍스트/게임 상태 등)을 받아 행동을 출력하는 **VLA(Vision-Language-Action) 에이전트 모델** 연구 및 구현
> - Instruction-following 성능 향상을 위한 데이터 설계, 학습 레시피 수립, **실패 케이스 분석 및 개선**
> - **Real-time 제약을 고려한 모델링/추론/서빙 최적화** 및 성능 검증
> - 대규모 학습 데이터 수집·정제 파이프라인 구축/운영
>
> **[지원 자격 (필수요건)]**
> - 인공지능(AI), 컴퓨터공학, 혹은 관련 기술 분야에서 현재 석사 혹은 박사 학위 과정을 밟고 있는 분
> - 연구를 주도할 수 있으며 딥러닝 관련 논문을 통해 빠르게 이해하고 실험 설계할 수 있는 능력
> - 새로운 domain에 빠르게 적응할 수 있는 분
> - 원활한 업무 커뮤니케이션 능력
>
> **[우대 요건]**
> - **AI/ML 탑티어 학회 (NeurIPS, ICLR, ICML 등) 관련 분야 탑티어 저널 논문 작성 경험자**
> - **Vision Language Action (VLA) Model 관련 탁월한 프로젝트 성과가 있으신 분**
> - 타 직군(게임 기획, 아트, 사업 등)과의 원활한 커뮤니케이션 스킬을 갖추고 있는 분
>
> **[전형 과정 및 제출 서류]**
> - 서류 전형 > 사전 인터뷰 (Phone Interview) > *직무 테스트 (Pre-Test) > 직무 면접 (Technical Fit Interview) > 종합 면접 (Culture Fit Interview) > 학/경력, 평판조회 > 처우협의 > 최종 합격 및 입사
> - **필요 서류**: 입사지원서 (CV 형식/가이드라인), 자기소개서(또는 커버레터), 경력기술서, 성적증명서, **포트폴리오 (필수)**

---

## 공고 ② `[AI Research Div.] AI Companion Team Internship (2년 이상 / 인턴)`
- **채용 링크**: [KRAFTON 채용공고 (job=4489)](https://www.krafton.com/careers/recruit-detail/?var_page=2&job=4489&search_list_cnt=10&search_keyword=ai%20research%20div) (`Greenhouse ID: 8125444002`)
- **근무지**: 역삼 센터필드
- **고용형태**: 인턴 (3개월, 정규직 전환 비대상) / 상시채용

### [공고 ①과의 차이점 요약]
- **미션**:
  - LLM, SLM, VLM, Human-AI Interaction 등 최신 AI 기술을 활용해 혁신적인 게임 플레이 경험 설계
  - 게임 환경에서 발생하는 실시간 상호작용과 대규모 동시 접속 환경에 대응하기 위해 텍스트 및 비주얼 입력을 처리하는 다양한 언어/멀티모달 모델의 연산 효율 최적화
  - 게임 디자이너, 엔지니어 등과 협업하여 실제 게임 프로젝트에 AI Companion 적용
  - 사용자 데이터 및 게임 로그 분석을 통해 모델 개선을 위한 피드백 루프 구축 및 지속 개선
- **필수요건**: AI/CS 관련 석·박사 과정 재학생, LLM/SLM/VLM 중 하나 이상 프로젝트 경험, PyTorch 활용 경험
- **우대요건**: 저명 학회(NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV, ACL, NAACL 등) 1저자 논문 게재·수상, vLLM/llama-cpp/SGLang 서빙 경험, LlamaIndex/LangGraph 활용 경험, 멀티모달 학습·추론 경험, 타 직군 커뮤니케이션 스킬

---

# 3. 1:1 매칭 팩트 시트

| 팀 요구사항 (공고 및 인터뷰) | 지원자 보유 실적 및 증빙 (CV & 포트폴리오) |
|:---|:---|
| **탑티어 학회 논문 (NeurIPS, ICML 등)** | • **NeurIPS 2025 Workshop (1저자)**<br>• **ICML 2026 Workshop (공동 1저자)**<br>• **ICCV 2025 Workshop (3저자 / 특허 2건)** |
| **VLA 및 멀티모달 에이전트** | • **2025 삼성 SCPC 23위/상위 9%**: Two-Stage VLM 경량화<br>• **PFVL**: 컴파일러 피드백 기반 자율 수정 에이전트 상태머신<br>• **Finance Jarvis**: FSM 및 가드레일 기반 멀티 에이전트 |
| **Real-time 제약 추론 최적화** | • 3B 미만 GPU 제약 환경: **Flash Attention v2, QLoRA, torch.compile 커널퓨전**으로 메모리 75% 절감 및 1.2배 가속 |
| **수학 기반 전용 아키텍처** | • Adele Ring 및 조화해석학(PFE) 기반 **Training-free 결정론적 텐서 임베딩** 설계 (소형 트랜스포머 정확도 66% $\to$ 92%) |
| **실패 케이스 분석 및 데이터 설계** | • **KCC 2024 (1저자)**: QA 환각 5대 유형 결정트리 및 메트릭 실패율 실증<br>• Jarvis 복기 대시보드(context_json) 기반 침묵 버그/손절 패턴 역추적 |

