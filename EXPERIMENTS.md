# EXPERIMENTS.md — 전체 실험 프로그램
Phase 1(측정): Exp1~3 핵심, Exp4 선택/후순위. Phase 2(정책): Go 이후.
모든 실험: CSV→results/csv/, 그림→results/figures/, 로그→results/logs/.
네이밍: results/csv/<exp>_<model>_<date>.csv. 헤더에 mlx-lm 버전 + 머신 상태 기록.

## Exp 1 — 비용 곡선: keep-and-read vs drop-and-recompute  [RQ1/H1]
목표: 두 primitive 비용 곡선을 컨텍스트 N에 대해 실측.
방법: 각 N(및 크기 sweep 모델)에 대해 —
  prefill_ms(N): N토큰 KV를 처음부터 recompute (fresh cache).
  decode_step_ms(N): N토큰 캐시 상태에서 decode 1스텝 (prefill 먼저 mx.eval!).
sweep: N∈[128,256,512,1024,2048,4096,8192]; 모델 0.5B/1.5B/3B/7B(4bit).
지표: prefill_ms, decode_step_ms(median/min/max), peak_mb.
분석: decode_ms가 N 따라 커지나(대역폭-바운드 시그니처)? "KV_bytes(N)/120GB/s"와 비교
  (KV_bytes ≈ 2·n_layers·n_kv_heads·head_dim·N·dtype_bytes). recompute는 N·모델크기에 어떻게 스케일?
  함의된 keep-vs-recompute crossover가 PCIe 예측/N≈50과 어떻게 다른가?
산출: results/figures/exp1_costcurve_<model>.png (prefill & decode vs N).

## Exp 2 — CPU/GPU 대역폭 경합 (decode 중)  [RQ2/H2]  ★새 측정, 핵심 노벨티
목표: GPU decode와 동시에 도는 CPU 메모리 트래픽이 공유 버스에서 decode throughput을 얼마나 떨어뜨리고
  N*을 어떻게 미는지 정량화.
방법: decode throughput(tok/s) + 실효 대역폭(powermetrics)을 (a) GPU decode 단독,
  (b) GPU decode + 통제된 CPU 대역폭 부하(STREAM류, 강도별, P-코어 핀) 조건에서 측정.
지표: decode tok/s, 실효 GB/s, (a) 대비 둔화, 경합 하 재측정 N*.
통제: CPU 부하 P-코어 핀, 강도 sweep, 발열 통제 특히 중요.
산출: 경합 곡선(decode tok/s vs CPU 부하 강도) + N* 이동 표.
노트(writeup, 2026-06-21): 경합 비대칭이 논문의 가장 깨끗한 노벨티 → writeup은 메커니즘(powermetrics A/B, 발열 아님) + recompute 면역 비대칭 + 정책 레버를 앞세울 것. [실측 완료: decode −36~38%; prefill 면역(−0.5~+7.5%); N*가 recompute 쪽 ~1.3–1.4× 이동.]

## Exp 3 — 정책 격차: 현행 프레임워크 vs 오라클  [RQ3/H3]  (정책 동기) [재정의 2026-06-21]
목표: 현행 프레임워크가 *재유도된 UMA 결정 구조*(offload 붕괴 + 경합 축)를 무시해서 흘리는 양을 정량화 → UMA-native 정책의 동기.
방법: 멀티턴 트레이스(ShareGPT/LMSys-Chat) 재생 — (a) mlx-lm 기본 rotating; (b) full-keep; (c) always-recompute;
  (d) 오라클 = Exp1/2 측정 비용모델로 매 결정 최저비용 선택(=상한).
지표: TTFT, inter-token latency, peak memory, throughput, OOM/크래시 발생률. 기본↔오라클 격차 = 정책이 회수할 헤드룸.
범위: **측정까지만. 정책 *구현*은 Phase 2. 시스템 구현으로 새지 말 것.**
산출: 정책별 비교 표 + 격차 플롯.

### online causal 휴리스틱 (Exp3, 측정 전용) [추가 2026-06-22]
목적: 예지 없는 단순 정책이 oracle 격차의 일부를 회수함을 입증(달성 가능성). oracle과 동일 비용 함수, 입력만 다름.
eviction 트리거(메모리 예산 초과로 KV 비워야 할 때) 시, 각 후보 시퀀스 점수:
  drop_gain(seq) = keep_cost(seq) − P_reuse(seq) × recompute_cost(seq)
  - keep_cost      = KV 점유 메모리 (2·layers·kv_heads·head_dim·N·dtype_bytes)
  - recompute_cost = Exp1 prefill(N) 비용곡선 룩업
  - P_reuse        = 최근성 근사. 첫 버전: 계단함수(마지막 사용 ≤ K턴 → 높음, 아니면 낮음). K 작은 sweep(2/4/8).
drop_gain 큰 순서로 drop, 예산 충족까지.
대조군(최종): rotating(mlx-lm 기본) / full_keep / always_recompute / oracle(상한) / causal(이 정책). [+LRU = causal의 recency-only 조상, 참조용]
지표(동일): TTFT, inter-token latency, peak memory, throughput, OOM/크래시율.
핵심 산출: **회수율 = causal이 oracle 격차의 몇 %를 회수하나** + rotating/full_keep/always_recompute 대비 우위.
주의: 첫 버전 정교화 금지. recency는 계단함수로 충분. 학습/예측 기반 확장은 Phase2.
검증: oracle과 causal이 *정확히 같은 비용 함수* 사용(차이는 P_reuse 입력뿐); recompute_cost가 Exp1 prefill(N) 곡선을 룩업.

### vllm-metal 비교군 추가 [2026-06-22]
최종 대조군에 vllm-metal(또는 vllm-mlx) paged-KV + 기본 eviction(LRU/recompute-default)을 같은 멀티턴 트레이스에서 *실측*(실 엔진 행동 관찰)해 추가 — "SOTA 서빙 엔진조차 단일 풀 비용 구조를 무시해 흘린다"(mlx-lm rotating보다 강한 baseline).
최종 대조군: rotating(mlx-lm) / full_keep / always_recompute / **vllm-metal paged+LRU** / oracle(상한) / causal(우리). [+lru = 시뮬레이션상 vLLM eviction의 proxy, 이미 포함]
주의: 실 엔진 측정만(Phase 2b 구현과 별개). 설치·실행이 무거우면 우선순위 낮춰 실 트레이스 Exp3(mlx 정책+oracle+causal) 먼저 완성하고 vllm-metal 비교군은 그다음.

### Exp3 보고 [갱신 2026-06-22, 승인]
- 헤드라인 = recovery-vs-K **곡선**(단일 K 아님). 본문 강조 K=8~16.
- baseline 표기: lru = "현행 엔진의 recency-기반 eviction을 *대표*"(vLLM proxy로 단정 금지).
- formulation(승인): oracle P_reuse=ever-reused(이진), causal P_reuse=exp(−idle/K), 동일 drop_gain=keep_ms−P_reuse×prefill(N). 유도 과정(naive 점수 실패→이 유도)을 방법론으로 명시. (근거 DECISIONS 2026-06-22 ★)

## Exp 4 — 크로스아키 대조점  [RQ4]  (선택, Phase 2 즈음)
목표: crossover가 통합 메모리 스펙트럼을 따라 *이동*함을 보임(풀 시스템 이식 X, 곡선만).
방법: Exp1 비용 곡선을 클라우드 분리형 GPU 한 대(PCIe; A10/4090)에서 재현 → N*_PCIe.
  M4 N*_UMA와 대조. GH200/GB200은 분석적으로 위치(인용, 실행 X).
산출: 대조 그림 한 장(N* across PCIe vs 단일 풀 UMA).

## Phase 2 — 적응형 정책  (Go 이후)
목표: mlx-lm KVCache 경로에 적응형 keep/recompute/evict 정책 구현. 측정된 UMA 비용 모델 기반
  (PCIe 항 없음; 대역폭-경합 + 연산 항). baseline 이기고 OOM/패닉 방지.
baseline: rotating, LRU, full-keep, always-recompute (+ Exp3 오라클을 상한으로).
지표: Exp3 + 정책 자체 오버헤드(가벼워야 함).
범위: 단일 풀 UMA; 멀티턴/agentic(진짜 통증); 긴 컨텍스트; 발열 효과.

### Phase 2a [신규 명세 2026-06-22 — mlx-lm 실 구현]
목표: Exp3 시뮬로 검증한 causal 정책을 *실제 mlx-lm KVCache에 구현*해 진짜 모델 돌리며 재현 → "정책이 실제 동작"(워크샵 격상: 측정 → 측정+정책).
방법: mlx-lm KVCache/RotatingKVCache 상속·후킹 → eviction 시점에 drop_gain 점수로 victim 선택(시뮬과 동일 공식). 실 멀티턴(make_prompt_cache prefix 재사용) 돌리며 측정.
baseline: rotating(mlx-lm 기본)/full_keep/always_recompute/causal(우리). 지표: TTFT·ITL·peak mem·throughput·OOM. 시뮬 recovery%와 실측 비교.
난이도: 중간(이식; 알고리즘 검증됨). 마찰=실 멀티턴 트리거·16GB OOM 통제·시뮬-실측 갭 해석.
off-ramp: 실측이 시뮬과 크게 어긋나거나 16GB 벽이면 그것도 발견(시뮬-실측 갭/메모리 제약 본질적)으로 기록, Phase 1만으로 워크샵 제출. 벽에 머리 박지 말 것.
주의: Phase 2a=단일 사용자·순차 맥락. 배칭/실 서빙 엔진 검증은 Phase 2b(vllm-metal, 랩 조건부).

### Phase 2a 워크로드 설계 [2026-06-22 확정]
구현(후킹, #4 정찰): mlx-lm LRUPromptCache 서브클래스, victim 선택(_lru.pop)을 drop_gain으로 교체. drop_gain(seq)=keep_ms−P_reuse×prefill_ms(N), 입력 모두 캐시 엔트리에 존재(nbytes→keep_ms via 대역폭, len(tokens)→recompute=Exp1 prefill 룩업, LRU 위치→recency→P_reuse=exp(−idle/K)). baseline=미변경 mlx-lm LRUPromptCache(실제 코드, proxy 아님). 가능하면 oracle 실측 변형(ever-reused) 추가.
워크로드: Exp3 ShareGPT 트레이스를 ≤8k 토큰 필터(극단꼬리 제외=통제) + 수십~수백 대화. 압박=동시성(KV 예산 공유)+예산 상한 작게(1~2GB) 인위 설정 — **절대 16GB 한계까지 밀지 말 것**(예산을 작게 걸어 통제 압박). 모델=1.5B-4bit 고정.
측정: TTFT·ITL·peak mem·OOM·throughput. baseline vs ours(vs oracle). 핵심=실측 회수율 vs 시뮬(50~92%). K=8~16. 통제: 1.5B 단독(GPU 다른 작업 금지=Exp2 오염 방지)·mx.eval·조건 인터리브+cooldown.
환경 통제(16GB 필수): 실험 전 무거운 앱(브라우저/Slack/Docker) 종료, memory_pressure free%(이상 ≥60%) 확인·기록(재현성).
정직한 단서: Phase 2a는 16GB 제약상 Exp3 부분집합(길이필터·통제예산·작은샘플). 절대 숫자는 Exp3와 다를 수 있으나 같은 샘플·압박서 baseline↔ours↔oracle 상대 회수율로 시뮬 검증.
off-ramp: 실측≠시뮬 크게 어긋나면 "시뮬-실측 갭"을 발견으로 기록·Phase 1 제출. OOM 통제 불가/재현 안 되면 "16GB 실구동 측정 한계" 기록·시뮬(Exp3) 주결과 유지. 어느 쪽이든 발견 전환, Phase 1 안전판.
