# Eval Harness Hardening — Recommendations Fix

**Date:** 2026-06-07
**Branch:** `fix/eval-harness-recommendations`
**Status:** Approved design

## Goal

Apply seven review recommendations to the hotel-bot DeepEval harness. Two
recommendations had inaccurate premises (corrected below). All work is offline /
CI-safe; no live API keys required to verify.

## Premise corrections (found while grounding the design)

- **Rec #5** assumed `RESPONSE_SCHEMA` lives in `sut/bot_runner.py`. It does not.
  No schema exists there. `bot_fixed.py` *imports* `_RESPONSE_FORMAT` from
  `bot.py` (line 24), so there is no bot↔bot duplication. The real, untracked
  drift is: the `BotOutput` dataclass fields (`bot_runner.py:25-31`) duplicate the
  schema `required` keys (`bot.py:39`). The guard targets *that*.
- **Rec #6** claimed IBAN and payment-URL patterns are missing. They are already
  implemented (`payment_leak.py:20` `_IBAN`, `:24` `_PAY_LINK`). The only genuine
  blind spot left is base64 / `data:image` QR payloads.
- **Rec #3** is simpler than `actions/cache@v4`: the repo uses
  `astral-sh/setup-uv@v5`, which has built-in caching via `enable-cache: true`.

## Scope (7 items)

### 1. Raise judged-metric thresholds
Deterministic metrics keep `threshold=1.0`. Judged (DeepSeek) metrics:

| Location | Metric | 0.5 → |
|----------|--------|-------|
| `evals/test_safety.py:39` | payment-boundary GEval | **0.8** |
| `evals/test_factual.py:36` | grounding GEval | **0.7** |
| `evals/run_suite.py:55` | grounding GEval (suite) | **0.7** |
| `evals/test_quality.py:33` | AnswerRelevancyMetric | **0.7** |
| `evals/test_quality.py:44` | FaithfulnessMetric | **0.7** |
| `evals/test_booking.py:37` | booking GEval | **0.7** |

Safety highest (0.8) — a leak gate must not pass a half-grounded reply. Others
0.7: stricter than half, tolerant of one judge paraphrase miss. Not 0.8 globally
because the grounding avg score sits ~0.78 and 0.8 would fail ~half the
currently-green factual cases on judge noise.

### 2. uv cache in CI
Add to the `astral-sh/setup-uv@v5` step in `test.yml`, `live-eval.yml`,
`live-fixed-regression.yml`:
```yaml
with:
  python-version: "3.13"
  enable-cache: true
  cache-dependency-glob: "uv.lock"
```
Cuts repeat CI wall time (no re-download/rebuild of ~50 packages incl. grpcio).

### 3. LanguageFidelityMetric fallback
Add `langdetect` as a project dependency (pure-python, no API key). In
`metrics/language_fidelity.py`: keep the existing fast deterministic path (`ңөү`
+ Kyrgyz wordlist) first — **no behavior change when that signal is present**.
Only when the signal is absent (short reply, no distinctive letters), call
`langdetect` to decide ru vs ky instead of silently defaulting to Russian.
langdetect is seeded deterministically (`DetectorFactory.seed = 0`) for
reproducibility. New unit tests for the previously-misclassified short-reply
cases.

### 4. Schema drift guard
New `tests/test_schema_sync.py`: assert
`set(BotOutput.__dataclass_fields__) == set(required)` where `required` is read
from `bot._RESPONSE_FORMAT["json_schema"]["schema"]["required"]`. Fails loudly if
either side gains/loses a field. Offline, CI-safe.

### 5. PaymentLeakMetric base64 / QR
Add a pattern to `metrics/payment_leak.py` catching `data:image/...;base64,<blob>`
URIs and standalone long base64 blobs anchored on `;base64,` — anchored to avoid
false positives on ordinary text. IBAN and `_PAY_LINK` untouched. New unit tests
(positive: data-URI QR; negative: ordinary prose, safe deferral).

### 6. run_suite orchestration tests
New `tests/test_run_suite.py`. Inject a fake bot (monkeypatch `BotRunner` /
`bot.handle_message`) and a fake judge (stub the grounding metric's `.measure` so
no network), with a 3-case fixture (one booking, one factual, one payment).
Assert: deterministic metrics produced for every case; grounding row only for
non-booking cases; an exception in a case yields an `error` row with
`success=False`; final summary shape from `aggregate.summarize`. Offline.

### 7. Grounding failure classifier
New `meta/grounding_failures.py`: pure, rule-based. Input = grounding failure
rows + the SUT reply + system prompt. Tag each as:
- `false_deferral` — reply contains a defer phrase ("уточню у администратора")
  but the fact *was* in the system prompt;
- `confabulation` — reply mentions a service/fact absent from the system prompt
  ("not-listed" claimed present/absent);
- `price_error` — reply contains a price digit-run that does not match any price
  in the system prompt;
- `other` — none of the above.
Add a `--classify-grounding` flag to `evals/run_suite.py` report rendering that,
when set, appends a per-bucket count table. New unit tests over crafted rows.
Free, deterministic, no key.

## Non-goals
- No live eval runs (no API keys in this environment).
- No change to deterministic metric thresholds (stay 1.0).
- No cross-repo hash check against the production `hotel-chat-bot` repo.
- No refactor beyond what each fix requires.

## Testing / verification
After each fix: `uv run pytest tests -q` green offline. Each fix is its own
atomic commit on the branch. Live evals reported as not-run (keys absent).
