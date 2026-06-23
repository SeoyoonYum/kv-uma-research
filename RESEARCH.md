# RESEARCH.md — 마스터 컨텍스트 (source of truth)
> Claude Code: 매 세션 이걸 먼저 읽어라. 프로젝트의 단일 진실 소스다.
> 진행되면 STATUS를 갱신하라. 목표/가설에서 벗어나는 변경은 반드시 먼저 플래그하라.

## 1. 한 줄
단일 풀 통합 메모리(Apple Silicon)에서의 KV 캐시 *결정* 정책(keep / recompute / evict).
PCIe 시대 비용 모델이 깨지는 좌표.

## 2. 목표 & 베뉴
- KAIST 학부생, 솔로 시작, ~1년, 타겟 = MLSys (워크샵 EuroMLSys/MLArchSys 먼저 → 메인 트랙).
- 기여 유형: 새 하드웨어 좌표(통합 메모리) × 현행 정책 대비 구체적 개선.
- Phase 1 = 측정(비용 구조가 다름을 증명). Phase 2 = 정책(현행 이김). 워크샵은 Phase 1로.

## 3. 핵심 가설 (지적 중심)
PCIe 분리형 비용 모델(FlexGen LP, vLLM swap-vs-recompute)은 PCIe 티어 간 전송을 전제.
단일 풀 UMA(Apple Silicon)에선 CPU·GPU가 한 메모리를 한 대역폭으로 공유, 티어 간 전송 없음. 따라서:
- PCIe 문헌의 지배 레버 "CPU로 offload"가 *퇴화*(같은 풀로 옮겨 GPU 가용 용량 안 생김).
  결정이 **keep-and-read vs drop-and-recompute**로 환원(+ 아주 긴 컨텍스트만 disk).
- keep-vs-recompute crossover N*이 PCIe 예측 및 하드웨어-무관 N≈50(KV-Direct)과 다르다.
- 공유 버스의 CPU/GPU 대역폭 *경합*(Apple Silicon에서 미측정)이 N*을 더 이동시킨다.

## 4. 포지셔닝 / 프레이밍 [주제 피벗 2026-06-23 #2 — 발견 2가 논문 메인]
**주제 피벗:** "KV-cache decision policy" 논문 → **"단일 풀 UMA의 CPU/GPU 경합 비대칭 측정 연구 + KV 함의"** 논문. 무게중심 = KV-policy → single-pool UMA contention asymmetry (measurement study). 더 systems/measurement, 덜 policy. (이유: KV-policy 프레이밍은 범위 넓고 발견1/3 약함 + positive policy 없어 리뷰어 아쉬움; 발견 2 중심이 더 날카롭고 덜 obvious, systems 리뷰어 관심 큼.)

NEW thesis: In single-pool UMA, external CPU memory traffic creates an *asymmetric interference channel* for LLM inference — decode is bandwidth-arbitration sensitive while prefill/recompute is comparatively compute-bound. This asymmetry changes the cost model for KV-cache management and makes static PCIe-era policies insufficient.

작업 제목: "When CPU Traffic Changes KV-Cache Decisions: A Measurement Study of Unified-Memory LLM Inference"

층 구조 (피벗 후):
- [메인 = 논문의 "한 방"] 발견 2 — 외부 CPU 메모리 트래픽 ↔ GPU decode의 단일 UMA 버스 경합 비대칭: decode ~38–44% 둔화 / prefill·recompute 면역 / 0.5–7B robust / GPU 클럭 유지 → 발열 아닌 arbitration. (RQ2/Exp2)
- [setup = 배경, 짧게] 발견 1 — decode=read-heavy, prefill=compute-heavy; UMA엔 offload 티어 없어 eviction=강제 recompute. *이제 메인 아니라 발견 2 이해용 배경.* Agent Memory(2603.04428) 인용.
- [decision implication] 발견 2 → KV: 경합이 keep-vs-recompute 상대 비용을 바꿔 decision boundary를 recompute 쪽으로 shift. **정직하게: 오늘 측정 모델에선 완전 역전 아닐 수 있음 → "boundary가 shift하므로 모델링돼야 한다"로 안전하게 쓰고, α-sweep + 경합 A/B 데이터로 "결정이 갈린다"를 받침.**
- [policy lesson = future, 짧게] 발견 3 — causal 단순 휴리스틱 실패 → contention-aware + reuse-prediction 필요.

현실성 프레이밍 (발견 2 아킬레스건 방어, 중요): "LLM 전용" 기기면 CPU 한가 → 경합 드묾(한계 인정). 그러나 on-device LLM은 보통 에이전트 도구·RAG·OS·전처리와 메모리를 *공유*하며, 단일 기기 공유야말로 UMA의 *기본 운영 양식*(데이터센터 GPU-전용 가정과 대비). → 경합은 코너케이스 아니라 이 하드웨어의 자연 조건.

기여 (피벗 후 순서):
- **(메인) 외부 CPU↔GPU decode 단일-UMA-버스 경합 비대칭의 첫 측정 + 메커니즘(arbitration evidence) + 크기 robust.**
- 그 비대칭이 KV 관리 비용 모델을 바꿈을 보임(decision boundary shift; α-sweep + 경합 A/B로 "결정 갈림" + contention-aware 정책 한 점=positive).
- 배경: 단일 풀 UMA에서 offload 퇴화 → eviction=recompute (Agent Memory와 함께 확립; 우리 정밀화 329→1587×).
- design lesson: 단순 recency 휴리스틱 한계 → contention-aware + 예측 필요(future).

베뉴: ML for Systems @ NeurIPS (4쪽 extended abstract, non-archival) → arXiv 선공개 + 메인 재제출 가능.

## 4b. 노벨티 / 관련연구 델타 [피벗 2026-06-23 #2 — 발견 2 distinction 못박기]
novelty는 "비대칭" 자체가 아님(prefill/decode 경합 = red ocean). 정확한 distinction (리뷰어 방어 핵심):

| 기존 interference 논문 | 우리 |
|---|---|
| GPU 내부 prefill↔decode 간섭 | 외부 CPU workload가 GPU decode 간섭 |
| datacenter GPU serving | on-device 단일 풀 UMA |
| GPU 내부 scheduling / SM 분할 | CPU/GPU 간 memory-fabric arbitration |
| prefill/decode co-location | agent / tool / RAG / OS co-execution |

대조 인용(intro에서 명시): Nexus(2507.06608)·DuetServe(2511.04791) = GPU-internal, 다른 세팅. multi-GPU CPU 병목(2603.22774) = 제어경로(대역폭 아님). Apple Silicon 프로파일(POMACS 2508.08531) = 단일 워크로드.
✅ 우리 것: 외부(비-LLM) CPU 메모리 트래픽 ↔ GPU decode가 단일 UMA 버스 경합 → decode ~38–44% 둔화·prefill 면역 → KV 복구 결정 함의. 조합(경합 주체 × 단일 풀 × KV 결정)이 미발표.
⚠ arbitration 표현 주의(Apple 메모리 컨트롤러 closed-source): "we prove arbitration mechanism" 금지 → **"evidence is consistent with shared-bandwidth arbitration rather than thermal throttling"** (powermetrics: GPU 클럭 유지·전력↑·throughput↓).
- 전제 우군: Agent Memory(2603.04428, UMA evict=recompute), When Quant Is Free(2605.05699), DBMS preemption(2411.07447).
- 발견 3 동기(데이터센터 선점): SAECache(2605.18825), KVCache-in-Wild(2506.02634), RLT/LBGR(2601.18999).

## 5. 하드웨어 (실험 플랫폼이자 연구 대상)
- M4 MacBook Air, 16GB 통합 메모리, 10코어(4P+6E), ~120GB/s, 팬리스.
- MLX + mlx-lm. 모든 실험은 *반드시 이 머신에서* — 하드웨어가 곧 측정 대상.
- 모델(4bit): 워크호스 Qwen2.5-1.5B; 크기 sweep 0.5B/1.5B/3B/7-8B. 13B+ 16GB 불가.

## 6. 절대 함정 3개 (어기면 결과는 쓰레기)
1. MLX는 LAZY → 모든 타이밍 mx.eval()로 강제 실행+동기화. decode 측정 시 prefill을 먼저 mx.eval해
   캐시 빌드 비용이 스텝 타이밍에 새지 않게.
2. 팬리스 발열 스로틀링 → warmup + N회 median + 포인트 간 cooldown; powermetrics로 온도/클럭;
   decode_max ≫ median이면 스로틀링.
3. 16GB 빠듯 → mx.set_memory_limit(~12GB); mlx_lm.server(75% wire) 금지; 앱 닫기; 작은 모델.
   (OOM/커널패닉은 연구 대상 실패모드 — 의도적으로 유도, 사고로 맞지 말 것.)
+ 전원 상태 고정(충전기), 백그라운드 정리, warm/cold 분리, CPU 부하는 P-코어 핀.
+ mlx/mlx-lm API는 버전마다 다름 — 매번 설치 버전으로 확인.

## 7. STATUS (갱신할 것)
- [x] 환경 셋업 — mlx 0.31.2 / mlx-lm 0.31.3 / numpy 2.4.6 / matplotlib 3.11.0 / Python 3.13; venv=./.venv.
      (API 확인: make_prompt_cache←mlx_lm.models.cache; set_memory_limit/get_peak_memory/reset_peak_memory/clear_cache 모두 top-level;
       mlx Dtype에 .itemsize 없음 → arr.nbytes//arr.size로 도출.)
- [x] Exp 1 첫 실행 (Qwen2.5-1.5B) — decode_ms는 N 따라 완만히 증가(10.7→15.2ms, ×1.43). floor ~11ms = per-step 가중치 읽기
      (120GB/s 대역폭 모델과 일치 ✓). N-항(KV 읽기)은 실효 ~62GB/s = 피크의 절반. prefill 122→12,000ms(초선형, O(N²) attention).
- [x] Exp 1 풀 sweep (크기 0.5/1.5/3/7B 4bit) — decode floor·recompute tax·UMA-vs-PCIe 격차가 모델 크기로 단조 증가.
- [x] 크로스오버 분석 — keep-and-read가 drop-and-recompute를 모든 N에서 압도(B*≈0.02–0.04 GB/s ≪ PCIe·UMA 대역폭).
      UMA-vs-PCIe 차이는 *eviction 경로*(메모리 압박)에 있다: PCIe는 KV를 CPU로 swap(~19ms, 보존), UMA는 swap 탈출구가 없어
      drop+recompute 강제(~12s) ≈ 600×@8192(1.5B); 크기로 ~1,587×(7B@4096)까지 확대. (recompute=실측, PCIe swap=모델값.)
- [x] Exp 2 (경합) — CPU 대역폭 부하가 decode를 ~36–38% 느리게(~69GB/s, 단조 증가). powermetrics A/B로 대역폭/중재 경합 확정(GPU 클럭 유지·전력↑인데 throughput↓ = 메모리 stall, 전력·발열 throttle 아님). 첫 sweep 버그(측정창 직전 prefill이 부하 수명 잠식 → 과소측정) 정정. / [ ] Exp 3 (정책 격차)
- [x] Go/No-Go 판단: **GO** — H1 강(Exp1 ~600–1,587× 격차) + H2 강(Exp2 ~38% 경합, 메커니즘 확정). 게이트 두 조건 충족.
- 현재 열린 질문: (1) **N* 이동 [답함]** — prefill 경합 거의 무반응(−0.5~+7.5%, compute-bound) vs decode ~38%(bandwidth-bound) → 경합 시 N*가 recompute 쪽 이동(~1.3–1.4×). Exp1 ~600×엔 못 미쳐 refine. 정책 레버=버스 경합 시 recompute 선호(PCIe엔 없음). (2) KV 읽기 실효 대역폭이 큰 모델서 더 낮은 이유. (3) 7B@8192 클린 단일 점. (4) PCIe swap=모델값 — Exp4 실측 대조.
- [x] **Phase 1 측정 GO** + 프레이밍 확정(2026-06-21): ②경합 비대칭을 전면 empirical로, ①eviction은 folklore의 첫 통제 정량화로 포지셔닝.
- [ ] 마무리 측정: 7B@8192 클린 1점, 경합 곡선 크기 sweep(+큰모델 KV대역폭 비효율 미니조사).
- [~] Exp 3 (정책 격차) — 다중 시퀀스 eviction 모델(가변 think-time) + online causal 휴리스틱, **실 ShareGPT(5k convs) 측정**. oracle 14.96<causal<lru 15.71<always_recompute 16.46s TTFT. **causal(예지 없음)이 LRU→oracle 격차의 44–75%(K=8–32) 회수 → 달성 가능성 입증(강).** rotating(기본)은 실 multiturn서 ctx 31%만 유지. spec 2곳 정합성 수정(oracle=ever-reused 이진, causal=exp(−idle/K) smooth) **챗 OK 필요**. budget robustness 확인됨(2/6GB 모두 LRU→oracle 41–83% 회수). vllm-metal 비교군=실엔진 보류(Docker-Mac Metal 패스스루 불가; 시뮬 lru가 proxy).
- [x] Phase 2a (causal 실 구현) — `policy/causal_cache.py`: mlx-lm `LRUPromptCache` 서브클래스, eviction victim을 drop_gain으로 교체(baseline=미변경 LRUPromptCache). 실 1.5B×ShareGPT×KV budget에서 **causal이 LRU 미달(모든 K, K↑일수록 악화)** = 정직한 negative. 진단: keep_ms≪recompute → drop_gain이 작은(싸게 recompute) conv 축출 → 작은 메모리 해제 → eviction churn. 발견 3개: (a) 단순 recency+recompute 휴리스틱 한계, (b) 시뮬-실측 갭(비용모델 과대평가), (c) reuse 예측이 본질적(예지 oracle만 LRU 능가). → 결과 섹션에 포함.
- **[x] 노벨티 재편(2026-06-23) — 무게중심 이동.** 발견1(eviction=recompute) → *전제/background* 강등(Agent Memory 2603.04428 published 선점; 우군 인용, 우리 정밀화 329→1587×만 작게). **발견2(CPU↔GPU UMA 버스 경합 비대칭+KV함의) → 중심 노벨티 격상**(3회 검색 후 비어있음 확인; 단 "비대칭"은 레드오션, 새로움=조합). 발견3 → *동기/future*(SAECache 등 데이터센터 선점, 인용). 발견2 현실성 프레이밍 = on-device LLM은 에이전트/RAG/OS/전처리와 메모리 공유 → 단일기기 공유가 UMA 기본 양식 → 경합은 코너케이스 아님. (RESEARCH §4/§4b, RELATED_WORK, DECISIONS 반영.)
- **[x] 베뉴 확정 = ML for Systems @ NeurIPS** (4쪽 extended abstract, non-archival, ~8월말 마감 추정). non-archival → arXiv 선공개 + 메인 재제출 가능.
- **[x] 주제 피벗(2026-06-23 #2) — 발견 2가 논문 메인.** "KV-cache decision policy" → **"단일 풀 UMA의 CPU/GPU 경합 비대칭 측정 연구 + KV 함의"**(measurement study). NEW thesis/제목 확정(§4). 발견1=setup, 발견3=design lesson/future. 발견2 distinction 못박음(외부 CPU↔GPU vs GPU-internal Nexus/DuetServe; arbitration="consistent with" 약하게). 퀄리티 우선(시간 압박 완화) → 약한 고리 3개 메우기 = EXPERIMENTS [P1/P2/P3]. (RESEARCH §4/§4b, DECISIONS, EXPERIMENTS, paper/outline 반영.)
- [ ] **다음 핸드오프(실험, 사용자 충전기 꽂고 P1부터)** — [P1] α-weighted drop_gain α-sweep(곡선) + oracle PCIe-cost vs UMA-cost A/B(경합 有/無 두 조건, "결정 갈림") + contention-aware victim(positive 한 점). [P2] 현실 CPU 부하 다양화(memcpy/mmap/random/전처리류) + intensity sweep 정교화(0/10/20/40/60/70 GB/s). [P3] Mac mini 교차검증(나중). *이번 턴은 문서·spec만, 측정 코드·결과 미변경.*
- [ ] arXiv/Semantic Scholar 알림 설정. 제출 직전 노벨티 재검증.
- [ ] 랩 컨택(한동수/박경수) — GO+측정 들고.

## 8. Go/No-Go (de-risk 게이트)
계속: Exp1의 N*이 PCIe/N≈50 예측과 유의하게 다르거나, Exp2에 측정 가능한 경합 효과.
피벗: N*이 기존 예측과 같고 경합 효과 없으면 → 죽은 가설에 몇 달 붓지 말 것.

## 9. 작업 협약 (챗 vs 코드)
- Claude Code(이 리포, M4): 코드 작성/실행/디버깅, MLX API 버전 매칭, CSV+플롯 생성, 통제 구현.
- 전략 챗(별도): 결과 *해석*, 다음 실험 설계, 논문 프레이밍, Go/No-Go 판단.
- CSV/플롯 + 열린 질문은 전략 챗으로 가져가 해석.
