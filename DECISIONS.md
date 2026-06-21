# DECISIONS.md — 결정 로그 (날짜 / 결정 / 이유)
2026-06: 주제 = 단일 GPU/통합 메모리 KV 캐시 결정 정책. 데이터센터 KV-perf는 포화로 확인(3회 검증).
2026-06: 제외 — 서빙 보안(흥미 없음), 측정-온리(노벨티 부족), 데이터센터 GB200 피벗(접근 불가·붐빔).
2026-06: 프레이밍 = 통합 메모리 스펙트럼, 단일 풀 극한에 집중, NVIDIA는 동기+대조로만.
2026-06: 플랫폼 = M4 16GB. Phase 1 측정-퍼스트 + Go/No-Go 게이트.
2026-06-20: Exp1 prefill = BODY-ONLY 측정 확정(transformer body만; lm_head/vocab projection 제외). 이유: KV 재계산 원시비용에 vocab projection 무관 + N=8192서 logits (1,8192,vocab) ≈2.5GB 폭발 회피.
2026-06-20: 발열 로깅 — pmset CPU_Speed_Limit는 Apple Silicon 미지원(Intel 전용)으로 확인 → 측정 스프레드(max/median, throttle_ratio)를 무-sudo throttle 신호로 사용. powermetrics(sudo)는 Exp2용 opt-in(PowerMetricsLogger).
2026-06-20: 7B sweep은 N≤4096로 cap(cooldown 12s). 이유: 7B@8192 prefill ≈64s/forward로 팬리스 스로틀이 prefill을 가짜로 부풀려 UMA-vs-PCIe 격차를 왜곡 → 클린 점만 채택.
2026-06-20: 크로스오버 결론 — UMA-vs-PCIe 차이는 resident 경로가 아니라 *eviction 경로*(메모리 압박)에 존재. UMA는 swap 탈출구 부재로 recompute 강제 → 가설 지지, 격차가 모델 크기로 강화(395×→1,587×).
2026-06-21: 리포 구조화 — kv-uma-research/ 부트스트랩(RESEARCH/EXPERIMENTS/DECISIONS/RELATED_WORK + src/common 리팩터링). exp1 공용 로직을 common/{measure,thermal,models,workloads}로 추출, exp2/3/4는 스켈레톤. exp1 재실행(smoke) 검증 통과, 기존 측정 데이터는 results/로 이관.
