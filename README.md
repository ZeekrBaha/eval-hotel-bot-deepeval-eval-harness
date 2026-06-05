# Hotel Bot DeepEval Harness

An evaluation harness for the Hotel WhatsApp Bot — a gpt-4o-mini assistant that handles Russian/Kyrgyz guest queries, booking intake, and safety rules. The harness uses [DeepEval](https://deepeval.com) as the pytest-native evaluation framework and a DeepSeek LLM-as-judge.

---

## System Under Test

The SUT is a structured-output hotel bot (`gpt-4o-mini`) that:
- Responds in the guest's language (Russian or Kyrgyz — never mixed).
- Answers grounding questions strictly from a known system prompt (prices, address, amenities, check-in/out times).
- Refuses to reveal payment credentials; defers to the administrator.
- Collects four booking slots (name, check-in, check-out, guest count) one question at a time before confirming.
- Rejects off-topic requests politely.

---

## Why DeepEval

- **pytest-native** — evals run in the same `pytest tests` command as unit tests, so CI has a single command.
- **Multi-turn support** — `ConversationalGEval` / `ConversationalTestCase` / `Turn` evaluate the full booking dialogue, not just the last turn.
- **Pluggable judge** — `DeepEvalBaseLLM` lets us inject DeepSeek as the judge (out-of-family vs the gpt-4o-mini SUT) to avoid self-preference bias.
- **Offline gate** — three deterministic metrics run with zero API keys; the CI workflow never needs secrets.

---

## Metric Stack

### Deterministic (no key required)

| Metric | File | What it checks |
|--------|------|---------------|
| `LanguageFidelityMetric` | `metrics/language_fidelity.py` | Reply language matches query language (Kyrgyz ↔ Russian heuristic). |
| `PaymentLeakMetric` | `metrics/payment_leak.py` | Bot never emits card/account numbers (≥13-digit runs). Headline safety gate. |
| `SlotExtractionMetric` | `metrics/slot_extraction.py` | Extracted booking slots match golden expected values (partial check). |

### LLM-as-Judge (DeepSeek, requires `DEEPSEEK_API_KEY`)

| Metric | Eval file | What it checks |
|--------|-----------|---------------|
| `GEval("Grounding")` | `evals/test_factual.py` | Factual accuracy vs system prompt; correct absent-service / deferral behavior. |
| `GEval("Payment Boundary")` | `evals/test_safety.py` | Judged layer over the deterministic gate; catches paraphrased leaks. |
| `ConversationalGEval("Booking Gate")` | `evals/test_booking.py` | Multi-turn: confirm only when all 4 slots present; ask for missing slots otherwise. |

---

## DeepSeek as Out-of-Family Judge

The SUT is `gpt-4o-mini` (OpenAI). Using another OpenAI model as the judge introduces self-preference bias — the judge tends to rate OpenAI-style phrasing as correct. `DeepSeekJudge` wraps DeepSeek's `deepseek-chat` via its OpenAI-compatible endpoint to eliminate this.

`DeepSeekJudge` implements `DeepEvalBaseLLM` with `generate(prompt, schema=None)` and `a_generate`. With a pydantic schema it forces JSON mode; without it returns free-text — both paths are used by DeepEval's internal scoring.

---

## Judge Validation (the differentiator)

`meta/judge_validation.py` runs the DeepSeek judge over every golden case, collects its PASS/FAIL verdict, and computes **Cohen's κ against human labels** — split by language (RU vs KY). The key hypothesis: κ on Kyrgyz is measurably lower than on Russian because DeepSeek has weaker Kyrgyz capability. A low KY κ means the judge cannot be trusted on that subset and human review is required.

Results land in `results/judge_validation.json` (gitignored — kept local).

---

## Golden Set

22 hand-labeled cases in `data/goldens.jsonl`:
- 5 factual (prices, check-in/out, address, amenities)
- 3 absent-service
- 2 off-topic
- 3 payment-safety
- 3 booking-complete / 3 booking-incomplete
- 2 language-fidelity
- 1 factual (unknown info → defer)

All `human_pass: true` by design (the bot is expected to pass). Negative-labeled cases are recorded live when the bot actually fails and a human labels them `false`.

---

## Keys

| Variable | Used for |
|----------|---------|
| `OPENAI_API_KEY` | SUT — calls `gpt-4o-mini` to produce bot replies |
| `DEEPSEEK_API_KEY` | Judge — calls `deepseek-chat` to grade replies |

Copy `.env.example` to `.env` and fill both keys. The `.env` is gitignored.

---

## Run Order

### Offline (no keys — CI-safe)

```bash
pytest tests -q
```

Runs 33+ deterministic unit tests. Safe in CI; no secrets needed.

### Live evals (both keys required)

```bash
# 1. Confirm keys loaded
python -c "from conftest import has_key; print(has_key('OPENAI_API_KEY'), has_key('DEEPSEEK_API_KEY'))"

# 2. Run factual, safety, booking evals (SUT + judge calls)
pytest evals -v

# 3. Judge validation — κ overall + RU/KY split
mkdir -p results
python -m meta.judge_validation
# writes results/judge_validation.json and prints the κ table
```

Failures in `pytest evals` are **findings** (the bot failing a grading criterion), not test-harness bugs.

---

## Tech Stack

- Python 3.13
- deepeval 4.0.5
- openai (SDK, used for both the gpt-4o-mini SUT and the OpenAI-compatible DeepSeek endpoint)
- pytest 8.2.0
- python-dotenv

---

## Architecture

```
sut/          BotRunner (DB-free re-implementation), LLMClient protocol, FakeLLM, OpenAIChat
metrics/      Three deterministic BaseMetric subclasses (no key)
judge/        DeepSeekJudge wrapping DeepEvalBaseLLM
golden/       Golden dataclass + JSONL loader
evals/        Live eval tests (GEval, ConversationalGEval) — skip without keys
meta/         Cohen's κ + confusion matrix; judge_validation CLI
tests/        Offline unit tests for all of the above
data/         system_prompt.txt (fictional hotel, filled); goldens.jsonl (22 cases)
results/      judge_validation.json (gitignored, written by live run)
```
