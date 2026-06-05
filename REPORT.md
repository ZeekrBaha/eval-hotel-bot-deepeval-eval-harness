# Hotel Bot Evaluation Report

> **Status:** skeleton — numbers marked `_(filled after live run)_` are placeholders.
> Fill after running `pytest evals -v` and `python -m meta.judge_validation`.

---

## 1. Failure Modes Under Test

The two highest-risk failure modes in a hotel bot are:

1. **Payment leak** — the bot reveals card numbers, IBAN, QR payment data instead of deferring to the human administrator. Caught deterministically by `PaymentLeakMetric` (no judge needed) and as a judged second layer by `GEval("Payment Boundary")`.
2. **Confident hallucination** — the bot invents a fact (wrong price, fake amenity, invented transfer service) instead of either citing the system prompt or deferring with "Уточню у администратора." Caught by `GEval("Grounding")`.

Secondary: language-fidelity flip (Kyrgyz query answered in Russian or vice versa); off-topic engagement; premature booking confirmation.

---

## 2. Metric Stack Summary

| Metric | Type | Key required | Eval file |
|--------|------|-------------|-----------|
| `LanguageFidelityMetric` | Deterministic | No | `tests/` |
| `PaymentLeakMetric` | Deterministic | No | `evals/test_safety.py` |
| `SlotExtractionMetric` | Deterministic | No | `evals/test_booking.py` |
| `GEval("Grounding")` | LLM-as-judge (DeepSeek) | OPENAI + DEEPSEEK | `evals/test_factual.py` |
| `GEval("Payment Boundary")` | LLM-as-judge (DeepSeek) | OPENAI + DEEPSEEK | `evals/test_safety.py` |
| `ConversationalGEval("Booking Gate")` | LLM-as-judge (DeepSeek) | OPENAI + DEEPSEEK | `evals/test_booking.py` |

---

## 3. Judge Validation — Cohen's κ (Judge vs Human)

> deepeval 4.0.5 · judge: `deepseek-chat` · SUT: `gpt-4o-mini` · goldens: 22

| Subset | n | κ | Agreement |
|--------|---|---|-----------|
| Overall | 22 | _(filled after live run)_ | _(filled after live run)_ |
| Russian (ru) | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |
| Kyrgyz (ky) | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |

**Confusion matrix (overall):**

|  | Judge PASS | Judge FAIL |
|--|------------|------------|
| Human PASS | TP = _(filled after live run)_ | FN = _(filled after live run)_ |
| Human FAIL | FP = _(filled after live run)_ | TN = _(filled after live run)_ |

**Key finding:** _(filled after live run — expected: KY κ < RU κ due to weaker Kyrgyz capability in DeepSeek)_

Raw data: `results/judge_validation.json` (gitignored, kept local).

---

## 4. Pass/Fail Grid by Kind

> Results from `pytest evals -v`. Each cell = pass_count / total for that kind.

| Kind | n | Passed | Failed | Pass rate |
|------|---|--------|--------|-----------|
| factual | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |
| absent_service | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |
| offtopic | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |
| payment_safety | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |
| booking_complete | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |
| booking_incomplete | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |
| language | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ | _(filled after live run)_ |

---

## 5. Deterministic Findings

### Language-fidelity flips

Offline unit tests confirm the heuristic detector works on the test corpus. Live flip count (mismatched language in actual bot replies):

- RU queries answered in non-RU: _(filled after live run)_
- KY queries answered in non-KY: _(filled after live run)_

The heuristic uses Kyrgyz-specific Cyrillic (ң ө ү) vs Russian-only (ы э ъ щ) as the primary signal. Short inputs and loanwords are known blind spots.

### Payment leaks

Deterministic `PaymentLeakMetric` result across 3 payment-safety goldens:
- Leaks detected: _(filled after live run)_ (expected: 0)

If the bot is well-aligned on the "Реквизиты отправит администратор" rule, this should be 0 leaks. Any non-zero count is a critical finding.

---

## 6. Limitations

1. **Single SUT model** — only `gpt-4o-mini` tested. A fine-tuned or RAG-augmented variant may behave differently; re-run the full harness.
2. **Heuristic language detector** — `detect_lang` misclassifies short strings (< 3 words), mixed-language inputs, and transliterated text. The language-fidelity metric has an "unknown" fallback that passes rather than false-fails.
3. **22 goldens by design** — enough to find systematic failures but not for a statistically robust κ estimate. The plus-or-minus 95% CI on κ from n=22 is approximately plus-or-minus 0.2 depending on prevalence; treat the KY/RU split as directional, not definitive.
4. **DeepSeek schema scoring coarser than logprobs** — DeepEval's `GEval` originally used model logprobs for calibrated scoring. DeepSeek does not provide logprobs, so the judge falls back to JSON schema scoring. This is less calibrated but avoids the need for an OpenAI key on the judge side.
5. **No adversarial injection testing** — prompt-injection attacks are not yet in the golden set. Add them as a follow-up.

---

## 7. Close

The harness demonstrates that a full LLM-as-judge evaluation pipeline — including judge-vs-human validation with Cohen's κ split by language — can be built on top of DeepEval in under 600 lines of application code. The deterministic payment-leak gate is the most important test and runs with zero API cost in every CI run.

The κ-by-language analysis is the differentiator: if Kyrgyz κ is substantially below Russian κ, it shows the judge has degraded reliability on the minority language and that metric should be supplemented with human review for KY cases.
