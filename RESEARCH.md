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

## 4. 포지셔닝 / 프레이밍 [개정 2026-06-21]
논문 한 줄: 단일 풀 통합 메모리에선 KV 관리 *결정 구조*가 다르다 — offload 티어가 붕괴해 eviction이 강제 recompute가 되고(통제 측정 600~1600×, 모델 크기로 악화), PCIe엔 없던 새 축(CPU/GPU 버스 경합: decode −38%, recompute 면역)이 생긴다. 우리는 이를 처음으로 통제·메커니즘 수준에서 측정하고, PCIe 시대 비용 모델이 표현 못 하는 결정 모델로 재유도하며, 현행 프레임워크가 이를 무시함을 보인다.

도입부 5비트:
1. 동기: UMA는 떠오르는 패러다임(Apple 소비자 ↔ NVIDIA GH200/GB200). 지배 비용 모델(FlexGen LP, vLLM swap/recompute)은 PCIe 분리 티어 전제.
2. 갭: 실무자는 UMA 메모리 압박이 recompute를 강제함을 안다(folklore/블로그). 그러나 통제 측정·메커니즘·비용 모델·정책 함의가 없다. (folklore를 명시적으로 인정하고 시작)
3. 결과1(엄밀성): eviction-cost 붕괴를 체계적으로 측정 — 600~1600×, 모델 크기로 스케일. offload 옵션 퇴화.
4. 결과2(신규, 전면): 미측정 효과 측정 — CPU/GPU 버스 경합(−38%, powermetrics로 발열 아님 확정) + recompute 면역 비대칭 = PCIe엔 없는 새 결정 축.
5. 함의: 현행(rotating, recompute-default)은 이 구조 무시 → 정량 격차(Exp3) → UMA-native 정책 동기(Phase2).

기여:
- 단일 풀 UMA의 KV 결정 공간 재유도(offload 퇴화 → keep-vs-recompute 환원).
- eviction-as-recompute 비용의 첫 통제·모델스케일 측정(600~1600×, confound 분리).
- Apple Silicon decode 중 CPU/GPU 버스 경합의 첫 측정 + 메커니즘 + recompute 면역 비대칭(actionable 정책 레버).
- 현행 프레임워크가 이를 흘림(정책 격차).

folklore 위험 방어: 블로그/업계 논의 명시 인용 + "첫 체계적·통제·메커니즘 측정 + 비용 모델"로 포지셔닝; bare "비싸다" 대신 메커니즘+비용모델+경합 측정 앞세움; 모델 스케일링(329→1587×)·경합 비대칭을 비자명 결과로 강조.

## 4b. 노벨티 / 관련연구 델타
- folklore(인용): backend.ai / touchdown-labs / min.io 블로그(eviction→recompute, UMA working-set), arXiv 2605.05699(UMA bandwidth 역전, quant).
- 인접·대조: arXiv 2501.16909(NVIDIA GPU 내부 대역폭 간섭, 다른 setting), arXiv 2508.08531(Apple Silicon 단일워크로드 대역폭 프로파일), KVSwap 2511.11907(on-device disk offload), vllm-mlx 2601.19139(UMA 서빙, generic LRU), Learning-to-Evict 2602.10238(하드웨어-무관 RL eviction — 하드웨어-인지 피벗 시 경쟁).
- 우리 델타: 통제 측정 + CPU/GPU 버스 경합 비대칭 + no-PCIe 결정 모델 재유도.

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
- [~] Exp 3 (정책 격차) — 인프라 구축·검증(측정 전용 시뮬). 합성 트레이스 첫 결과: oracle만 무크래시+full ctx+최소지연; rotating ctx ~6% 손실+극한 OOM, full_keep OOM 압박따라↑(0→79%), always_recompute 지연 ~1.79× 과다. 실 ShareGPT/LMSys 트레이스로 magnitude 정련 남음.
- [ ] arXiv/Semantic Scholar 알림 설정. 제출 직전 노벨티 재검증.
- [ ] 랩 컨택(한동수/박경수) — GO+측정 들고.

## 8. Go/No-Go (de-risk 게이트)
계속: Exp1의 N*이 PCIe/N≈50 예측과 유의하게 다르거나, Exp2에 측정 가능한 경합 효과.
피벗: N*이 기존 예측과 같고 경합 효과 없으면 → 죽은 가설에 몇 달 붓지 말 것.

## 9. 작업 협약 (챗 vs 코드)
- Claude Code(이 리포, M4): 코드 작성/실행/디버깅, MLX API 버전 매칭, CSV+플롯 생성, 통제 구현.
- 전략 챗(별도): 결과 *해석*, 다음 실험 설계, 논문 프레이밍, Go/No-Go 판단.
- CSV/플롯 + 열린 질문은 전략 챗으로 가져가 해석.
