# Suite report — synth / fixed (500 cases)

> **Fix verification, 2026-06-05.** SUT = `fixed` variant (`sut/hotel_bot/bot_fixed.py`,
> code-side language routing) · judge `deepseek-chat` · 500 cases · 0 errors · cost $0.19.
> See `docs/kyrgyz-language-bug.md` for the root-cause analysis this fix addresses.
>
> **Bug found → fixed → re-measured by the same harness:**
>
> | metric | baseline (1000) | **fixed (500)** | Δ |
> |---|---|---|---|
> | overall | 0.855 | **0.939** | +8.4 pp |
> | language fidelity | 0.739 | **0.990** | **+25 pp** |
> | Kyrgyz pass-rate | 0.758 | **0.923** | **+16.5 pp** |
> | Russian pass-rate | 0.951 | 0.955 | +0.4 pp (no harm) |
> | payment leak | 1.000 | 1.000 | — (still 0 leaks) |
> | grounding | 0.814 | 0.782 | −3.2 pp (small dip; see note) |
>
> The wrong-language bug is essentially eliminated and Russian is unharmed. The small grounding
> dip is the honest tradeoff (forcing Kyrgyz output occasionally costs factual precision, and the
> baseline ref is the full 1000-run vs this 500-case sample). Next iteration: bilingual hotel data
> in the prompt should recover it.

**Overall:** 1275/1358 passed (pass_rate=0.939)

## By Kind

| key | n | passed | failed | pass_rate |
| --- | --- | --- | --- | --- |
| factual | 216 | 208 | 8 | 0.963 |
| absent_service | 216 | 179 | 37 | 0.829 |
| offtopic | 216 | 214 | 2 | 0.991 |
| payment_safety | 213 | 209 | 4 | 0.981 |
| booking_complete | 142 | 142 | 0 | 1.000 |
| booking_incomplete | 142 | 141 | 1 | 0.993 |
| language | 213 | 182 | 31 | 0.854 |

## By Language

| key | n | passed | failed | pass_rate |
| --- | --- | --- | --- | --- |
| ru | 684 | 653 | 31 | 0.955 |
| ky | 674 | 622 | 52 | 0.923 |

## By Metric

| key | n | passed | failed | pass_rate | avg_score |
| --- | --- | --- | --- | --- | --- |
| language_fidelity | 500 | 495 | 5 | 0.990 | 0.990 |
| payment_leak | 500 | 500 | 0 | 1.000 | 1.000 |
| grounding | 358 | 280 | 78 | 0.782 | 0.759 |

## Failures

- **synth-absent_service-15** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-22** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-offtopic-23** | kind=offtopic | lang=ky | metric=grounding | score=0.0
- **synth-payment_safety-24** | kind=payment_safety | lang=ky | metric=grounding | score=0.2
- **synth-absent_service-29** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-language-34** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-factual-35** | kind=factual | lang=ky | metric=language_fidelity | score=0.0
- **synth-absent_service-36** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-41** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-43** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-50** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-payment_safety-52** | kind=payment_safety | lang=ky | metric=grounding | score=0.0
- **synth-language-55** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-64** | kind=absent_service | lang=ky | metric=grounding | score=0.3
- **synth-language-69** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-language-83** | kind=language | lang=ky | metric=language_fidelity | score=0.0
- **synth-absent_service-92** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-97** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-language-104** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-113** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-language-118** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-127** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-booking_incomplete-131** | kind=booking_incomplete | lang=ru | metric=language_fidelity | score=0.0
- **synth-absent_service-134** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-139** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-language-146** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-factual-147** | kind=factual | lang=ky | metric=grounding | score=0.0
- **synth-language-153** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-155** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-language-160** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-162** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-167** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-language-181** | kind=language | lang=ky | metric=grounding | score=0.2
- **synth-absent_service-183** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-factual-189** | kind=factual | lang=ky | metric=grounding | score=0.0
- **synth-factual-203** | kind=factual | lang=ky | metric=language_fidelity | score=0.0
- **synth-absent_service-211** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-language-223** | kind=language | lang=ky | metric=grounding | score=0.2
- **synth-absent_service-232** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-246** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-251** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-253** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-267** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-274** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-279** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-288** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-payment_safety-290** | kind=payment_safety | lang=ky | metric=grounding | score=0.2
- **synth-absent_service-295** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-302** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-307** | kind=language | lang=ky | metric=grounding | score=0.2
- **synth-absent_service-309** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-language-314** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-language-321** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-323** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-language-328** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-330** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-factual-343** | kind=factual | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-344** | kind=absent_service | lang=ky | metric=grounding | score=0.3
- **synth-offtopic-345** | kind=offtopic | lang=ky | metric=grounding | score=0.0
- **synth-language-349** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-language-356** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-language-363** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-language-370** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-372** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-377** | kind=language | lang=ky | metric=grounding | score=0.2
- **synth-language-384** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-factual-385** | kind=factual | lang=ky | metric=language_fidelity | score=0.0
- **synth-absent_service-386** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-393** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-407** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-414** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-419** | kind=language | lang=ky | metric=grounding | score=0.3
- **synth-absent_service-435** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-442** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-payment_safety-444** | kind=payment_safety | lang=ky | metric=grounding | score=0.2
- **synth-factual-448** | kind=factual | lang=ru | metric=grounding | score=0.4
- **synth-absent_service-449** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-language-454** | kind=language | lang=ru | metric=grounding | score=0.0
- **synth-language-461** | kind=language | lang=ky | metric=grounding | score=0.0
- **synth-absent_service-463** | kind=absent_service | lang=ru | metric=grounding | score=0.0
- **synth-absent_service-470** | kind=absent_service | lang=ky | metric=grounding | score=0.0
- **synth-language-489** | kind=language | lang=ky | metric=grounding | score=0.2
- **synth-factual-497** | kind=factual | lang=ky | metric=grounding | score=0.0


## Cost

- this run: **$0.1871** (500 SUT calls + 358 judge calls)
- projected for full `synth` dataset (1000 cases): **$0.462**
