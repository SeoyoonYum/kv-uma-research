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
2026-06-21: 프레이밍 확정 — 논문 thesis = "단일 풀 UMA의 KV 관리 결정 구조를 *처음으로 통제·메커니즘 수준에서 측정*하고 결정 모델로 *재유도*". NOT "eviction이 비싸다는 발견"(그건 folklore).
2026-06-21: 노벨티 판정 (제출 전 게이트) —
  ① eviction=강제 recompute(600~1600×)는 정성적으로 folklore다. 다수 블로그(backend.ai, touchdown-labs, min.io)가 현상을 논하고, touchdown-labs는 16/32GB 통합 메모리 Mac + MLX로 우리 실험 셋업을 그대로 제안. arXiv 2605.05699 "When Quantization Is Free"는 UMA에서 bandwidth-driven 비용 역전을 보였으나 *quantization*에 대해서.
  ② Apple Silicon에서 decode 중 CPU/GPU 공유 버스 경합 측정 + recompute 면역 비대칭 = 미발표. 가장 가까운 학술 논문(arXiv 2501.16909)은 NVIDIA GPU 내부 멀티테넌시(다른 setting). → ②가 가장 깨끗한 노벨티.
  → 프레이밍 반전: ②(경합 비대칭)를 전면 empirical로, ①은 "folklore의 첫 통제 정량화 + 비자명한 모델-크기 스케일링(329→1587×)"으로 포지셔닝. folklore 소스를 명시적으로 인용하고 "첫 체계적·통제·메커니즘 측정 + 비용 모델"로 차별화.
2026-06-21: 결과 — 측정만으론 folklore-노출. Exp3(정책 격차)+Phase2(정책 구현) 중요도 상승. 정책이 "folklore 확인"을 "기여"로 전환.
2026-06-21: arXiv/Semantic Scholar 알림 설정 필요("unified memory KV cache", "Apple Silicon LLM inference", cs.DC/cs.LG). 제출 직전 노벨티 재검증(경쟁 가능 — 동일 셋업이 블로그에 공개됨).
2026-06-21: 베뉴 = EuroMLSys 1순위 / MLArchSys 2순위. Phase 1 = 측정 논문.
2026-06-22: Exp3 인프라 구축·검증(exp3_policygap.py + workloads.py; 측정 전용 시뮬, MLX 불필요 — Exp1 비용곡선을 비용모델로 채점). 합성 트레이스 + 동시성(메모리 압박) sweep 첫 결과: 현행 정책은 각각 ≥1축 실패 — rotating(mlx-lm 기본) ctx ~6% 무성 손실 + 극한 압박 OOM 79%; full_keep OOM 0→6→40→79%(압박↑); always_recompute 지연 ~1.79× 과다. **oracle만 0% OOM + full ctx + 최소 지연**(예산 초과 턴만 recompute) → full_keep↔always_recompute 보간. 격차 = UMA-native 적응 정책 헤드룸 → Phase2 동기. (합성 트레이스·offline 상한 oracle — 실 ShareGPT/LMSys로 magnitude 정련 남음.)
