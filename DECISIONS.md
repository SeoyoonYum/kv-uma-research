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
2026-06-21: Exp2(경합) — CPU 대역폭 부하(STREAM duty-cycle, P-코어 QoS)가 GPU decode를 ~36–38% 느리게(N=2048/8192, ~69 GB/s, 단조). 메커니즘 powermetrics A/B로 확정: 부하 시 GPU 클럭 유지/상승(1465MHz)·전력↑인데 throughput↓ → 대역폭/메모리 중재 경합, 전력·발열 throttle 아님. (M4 P-코어 1개가 ~92 GB/s 포화 → 강도축은 스레드수 대신 duty-cycle%.)
2026-06-21: Exp2 측정 버그 정정 — 첫 sweep(2.5–11.5%)은 측정창 직전 prefill이 CPU 부하 수명을 잠식(큰 N서 부하 조기 만료)해 과소측정. 캐시를 부하 전 빌드하도록 수정 → 정정값 ~36–38%, 프로브와 일치. ("짧은 컨텍스트가 더 민감"도 이 버그 아티팩트였음.)
2026-06-21: H2 판정 정정 — 버그값 근거 "경합 약함"은 철회. H2 강하게 지지(~38% 실효). 단 N* *이동*은 prefill-under-contention 미측정이라 미결(decode·prefill 둘 다 같은 버스 → 상대 민감도 필요). Go/No-Go = GO(H1 강 + H2 강).
2026-06-21: powermetrics passwordless sudo를 /usr/bin/powermetrics 한정으로 설정(/etc/sudoers.d/powermetrics). GPU 클럭/residency/전력 트레이스용(thermal.PowerMetricsLogger). 제거 가능.
2026-06-21: Exp2 N*-이동 닫음(exp2_prefill_contention.py) — prefill(recompute)은 경합에 거의 무반응(−0.5% @N2048, +7.5% @N4096; compute-bound, 가중치 N토큰 재사용·산술강도 높음)인데 decode(keep)는 ~37–39%(bandwidth-bound, 토큰마다 전체 가중치+KV 재독). 비대칭 → 경합 시 keep이 recompute 대비 ~1.3–1.4× 상대적 비쌈 → **N\*가 recompute 쪽으로 이동(directionally 확정)**. 정책 레버: 버스 경합 시 recompute 선호 — PCIe 모델에 없는 UMA 고유. 단 1.3–1.4×는 Exp1 ~600–1,587×에 비해 작아 지배 비대칭은 안 뒤집음(refine). H2 3다리(실재·대역폭·비대칭) 완성.
