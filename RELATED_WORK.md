# RELATED_WORK.md  (전체 주석은 deep-research 리포트에)
## Tier 1 핵심 (토대·대조)
- KV-Direct / Residual Stream (arXiv 2603.19664): recompute>read, 디코딩 bandwidth-bound, N≈50. 하드웨어-무관 → UMA로 특화할 메커니즘.
- When Quantization Is Free (2605.05699): KV quant tradeoff가 UMA에서 역전. "UMA서 비용 구조 바뀜"의 가장 가까운 증거.
- DBMS-Inspired Preemption (2411.07447): PCIe recompute>swap crossover + "end-device선 swap 유리" Remark(우리 punt).
## Tier 2 경쟁·인접 (주시)
- Learning to Evict from KV Cache (2602.10238, Apple): RL eviction, 하드웨어-무관. 하드웨어-인지 피벗 시 최대 위협.
- vllm-metal (Docker×vLLM 공식 플러그인, MLX 백엔드): 단일 풀 제로카피 + paged attention(experimental), v0.2.0 unified paged varlen Metal 커널(TTFT 83×↑). → PagedAttention 메커니즘 이식이지 비용 재유도 아님; KV 정책 vLLM 그대로(LRU/recompute). **Phase 2b 구현 타겟 후보 + Exp3 비교군.**
- vllm-mlx (2601.19139, EuroMLSys'26): UMA continuous batching + paged KV + prefix + SSD 티어링; eviction은 generic LRU.
- KVSwap (2511.11907): on-device disk-aware offload(디스크 I/O 바운드, in-memory 결정 아님).
메모: vllm-metal/vllm-mlx 둘 다 paged KV·단일 풀 인지는 했으나 비용 모델 재유도·CPU/GPU 버스 경합·UMA-native eviction 정책은 미탐 = 우리 델타. 릴리스 빠름 → watch list, 제출 직전 재검증.
## Tier 3 토대 계보 (읽음): FlexGen 2303.06865 / vLLM 2309.06180 / Orca(OSDI'22) / InfiniGen(OSDI'24)
## Tier 4 UMA 특성화: Profiling LLM on Apple Silicon 2508.08531 / Framework 비교 2511.05502 / Mobile SoC 경합 2501.14794
## Tier 5 데이터센터 UM(대조용): SuperInfer 2601.20309 / Oneiros 2507.11507 / Dyn KV Placement 2508.13231
