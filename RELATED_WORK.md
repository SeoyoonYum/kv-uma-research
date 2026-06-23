# RELATED_WORK.md  (전체 주석은 deep-research 리포트에. Tier 재편 2026-06-23 — 노벨티 무게중심 이동)

노벨티 무게중심: 발견1=전제(background), **발견2=중심 노벨티**, 발견3=동기(future).
각 항목 끝에 "우리 델타" 한 줄.

## Tier 1 — 전제·대조 핵심

### (a) 전제 우군 — 발견1(eviction=강제 recompute)을 *함께* 확립 (반박 아님)
- **Agent Memory Below the Prompt (2603.04428)**: UMA에서 evict=강제 recompute를 published로 확립. → 우리 델타: 반박 아닌 *우군 인용*; 우리는 통제 측정 + 모델-크기 스케일링(329→1587×)으로 정밀화(작게 남김).
- When Quantization Is Free (2605.05699): UMA bandwidth-driven 비용 역전(quant 축). → 우리 델타: 우리는 quant 아닌 eviction/경합 축.
- DBMS-Inspired Preemption (2411.07447): PCIe recompute>swap crossover + "end-device선 swap 유리" Remark. → 우리 델타: 단일 풀(swap 티어 부재)을 실측, end-device 가정을 측정으로 대체.

### (b) 발견2 대조 — *다른 세팅*, 명확히 구분해 인용 (혼동 차단)
- **Nexus (2507.06608) / DuetServe (2511.04791) / Sarathi / DistServe**: GPU 내부 prefill↔decode 단계 경합. → 우리 델타: 그건 *LLM 내부* 단계끼리(같은 GPU); 우리는 *외부 비-LLM CPU 트래픽* ↔ GPU decode가 단일 UMA 버스 경합. 경합 주체가 다름.
- CPU-induced multi-GPU slowdowns (2603.22774): multi-GPU에서 CPU 제어경로 병목(대역폭 아님). → 우리 델타: 다른 메커니즘(메모리 arbitration vs 제어경로) × 단일 UMA 풀.
- POMACS Apple Silicon profiling (2508.08531): UMA 대역폭 *단일* 워크로드 프로파일. → 우리 델타: 우리는 *동시* 워크로드 경합 + KV 복구 결정 함의.
- Embodied Edge survey (2603.16952): UMA arbitration 개념을 언급(서베이). → 우리 델타: 우리는 측정·메커니즘 확정(powermetrics)·정책 함의.

## Tier 2 — 발견3 동기 (데이터센터 선점 → 노벨티 아님, 반드시 인용)
- **SAECache (2605.18825) / KVCache-in-Wild (2506.02634) / RLT·LBGR (2601.18999)**: reuse 예측 기반 KV 정책·재사용 분석. → 우리 델타: 전부 데이터센터(PCIe); 우리는 "UMA에선 단순 recency 휴리스틱 부족 → reuse 예측이 본질적"을 측정으로 *동기화*(future work). 예측 정책 자체는 우리 기여 아님.

## Tier 2 — 인접 시스템 (UMA 서빙·offload)
- vllm-mlx (2601.19139, EuroMLSys'26): UMA continuous batching + paged KV + prefix + SSD 티어링; eviction은 generic LRU. → 우리 델타: 비용 모델 재유도·CPU/GPU 버스 경합·UMA-native eviction 미탐. (vllm-metal = PagedAttention *메커니즘* 이식, Phase 2b 구현 타겟 후보)
- KVSwap (2511.11907): on-device disk-aware offload(디스크 I/O 바운드). → 우리 델타: in-memory recompute-vs-read 결정 아님(우리는 단일 풀 내부 결정).
- Learning-to-Evict from KV Cache (2602.10238, Apple): RL eviction, 하드웨어-무관. → 우리 델타: 하드웨어-인지(UMA 비용·버스 경합) 피벗 시 직교/경쟁.

## Tier 3 — 토대 계보 (읽음)
KV-Direct / Residual Stream (2603.19664, recompute>read·decode bandwidth-bound·N≈50, HW-무관 → UMA 특화 메커니즘) / FlexGen 2303.06865 / vLLM 2309.06180 / Orca(OSDI'22) / InfiniGen(OSDI'24).

## Tier 4 — 데이터센터 UM / UMA 특성화 (대조용)
SuperInfer 2601.20309 / Oneiros 2507.11507 / Dyn KV Placement 2508.13231 / Framework 비교 2511.05502 / Mobile SoC 경합 2501.14794.

---
메모: vllm-metal/vllm-mlx 둘 다 paged KV·단일 풀 인지는 했으나 **비용 모델 재유도·CPU/GPU 버스 경합·UMA-native eviction 정책은 미탐 = 우리 델타.** 릴리스 빠름 → watch list, 제출 직전 재검증. 베뉴 = ML for Systems @ NeurIPS(4쪽, non-archival).
