# Hotel Bot — DeepEval Evaluation Harness

> **One sentence:** this repo evaluates a real bilingual (Russian/Kyrgyz) hotel WhatsApp
> bot using [DeepEval](https://deepeval.com), with a **DeepSeek LLM-as-judge that is itself
> validated** against human labels using Cohen's κ — and it finds that the bot is markedly
> weaker on Kyrgyz.

If you are reading this cold (including future-me): start with **§1 Mental model** and
**§2 Where the SUT lives**. Those two sections explain the whole repo.

---

## 1. Mental model — two halves

Every LLM evaluation has two separate things. Keep them straight and the rest is easy:

```
   ┌─────────────────────┐        ┌──────────────────────────────────┐
   │  SUT                 │        │  EVALUATOR (this harness)        │
   │  the hotel bot       │  reply │  - deterministic metrics (regex) │
   │  gpt-4o-mini         │ ─────► │  - LLM-as-judge (DeepSeek)       │
   │  (vendored, sut/)    │        │  - judge validation (Cohen's κ)  │
   └─────────────────────┘        └──────────────────────────────────┘
```

- **SUT = System Under Test = the hotel bot.** We do not write it here; we *evaluate* it.
  It is vendored into `sut/` (see §2).
- **Evaluator = everything else.** It feeds the bot golden test queries, grades the
  replies, and — crucially — **measures whether the grader itself can be trusted** (§5).

The "validate the grader, then trust the grader" loop is the point of the project, not the
pass/fail grid.

---

## 2. Where the SUT lives (read this — it's the #1 source of confusion)

The bot under test is **vendored** into this repo so the harness is self-contained:

| Path | What it is |
|------|-----------|
| `sut/hotel_bot/bot.py` | **The actual production bot**, copied byte-for-byte from `hotel-chat-bot/core/bot.py`. The ONLY change is the `db` import path. Default `gpt-4o-mini` temperature (nondeterministic — see §8). **Do not edit its logic**; keep it in sync with production. |
| `sut/hotel_bot/db.py` | An **in-memory** stand-in for the production Supabase db (same function signatures + two eval helpers `reset`/`set_history`). The production bot persists history/counters in Postgres; none of that is what we evaluate, so we swap it for dicts. |
| `sut/bot_runner.py` | `BotRunner` — a thin **driver**, not bot logic. `run(messages)` resets the db, seeds the scripted prior turns, sends the final user message through the real `handle_message`, and returns a `BotOutput`. |
| `sut/prompt.py` | `load_system_prompt()` — loads the hotel data so the **judge** can use it as grounding context. |
| `data/system_prompt.txt` | The bot's behavior contract (a *filled* fictional "Ала-Тоо" guesthouse: prices, address, amenities, RU/KY rules, payment boundary). This is the grounding ground-truth. |

There is **one** SUT, the vendored real bot. There is no re-implementation. The production
repo (`hotel-chat-bot`) is a *separate* project; this repo only borrows its `bot.py`.

---

## 3. What the bot does (the behavior we grade)

The hotel bot is a `gpt-4o-mini` assistant with structured JSON output
(`{reply, is_booking_intent, guest_name, check_in, check_out, num_guests}`) that must:

1. **Answer in the guest's language** — Kyrgyz query → Kyrgyz reply, Russian → Russian, never mixed.
2. **Stay grounded** — answer only from the system prompt; if the info isn't there, defer with "Уточню у администратора" (don't invent).
3. **Handle absent services correctly** — a service in the "Чего нет" list → say "no"; a service not listed at all → defer (don't claim it's absent).
4. **Never reveal payment details** — defer to the human admin.
5. **Collect booking slots one question at a time**, and confirm only when all four are present.
6. **Refuse off-topic** requests politely.

Each of these maps to a metric in §4.

---

## 4. The metric stack

Two layers. The deterministic layer needs no API key and runs in CI; the judged layer needs keys.

### Deterministic (no key, no judge)

| Metric | File | Checks | Live finding |
|--------|------|--------|--------------|
| `PaymentLeakMetric` | `metrics/payment_leak.py` | reply contains no card/account/QR digits (≥13-digit run) | **0 leaks** |
| `LanguageFidelityMetric` | `metrics/language_fidelity.py` (wired live in `evals/test_language.py`) | reply language matches query language (Kyrgyz `ң ө ү` + word list vs Russian) | **5/7 Kyrgyz replies came back in Russian** |
| `SlotExtractionMetric` | `metrics/slot_extraction.py` | extracted booking slots match the golden's expected values | — |

### LLM-as-judge (DeepSeek; needs `OPENAI_API_KEY` for the SUT + `DEEPSEEK_API_KEY` for the judge)

| Metric | Eval file | Checks |
|--------|-----------|--------|
| `GEval("Grounding")` | `evals/test_factual.py` | factual accuracy vs system prompt; correct absent-service / deferral behavior; off-topic refusal |
| `GEval("Payment Boundary")` | `evals/test_safety.py` | judged red-team layer over the deterministic payment gate |
| `ConversationalGEval("Booking Gate")` | `evals/test_booking.py` | multi-turn: confirm only when all 4 slots present, else ask for a missing one |
| `AnswerRelevancyMetric` | `evals/test_quality.py` | **was the answer helpful / on-topic?** |
| `FaithfulnessMetric` | `evals/test_quality.py` | **no hallucination** — every claim supported by the system prompt (passed as retrieval context) |

---

## 5. The differentiator — validate the judge with Cohen's κ

A judge you haven't measured is a judge you can't trust. `meta/judge_validation.py` computes
**Cohen's κ between the DeepSeek judge and human labels**, split by language.

**The subtlety that makes this real:** κ needs *variance* in the human labels (both pass and
fail). The golden set encodes only expected-correct behavior (every label "pass"), so κ over
it is **degenerate** — it collapses to 0 no matter how good the judge is. So validation runs
over a separate, **balanced, hand-labeled fixture**:

- `data/judge_validation_set.jsonl` — 16 *fixed* replies: 8 correct + 8 deliberately-wrong
  (hallucinated price, payment leak, wrong-language KY reply, false "we have a pool", premature
  booking confirm…), across both languages. The judge scores each; κ measures whether it tracks
  the human answer key.

```bash
python -m meta.judge_validation fixture   # → results/judge_validation_fixture.json
```

**Result: κ = 1.00 overall, 1.00 Russian, 1.00 Kyrgyz** (8 TP / 8 TN / 0 FP / 0 FN). The judge
is trustworthy in both languages, including catching every planted Kyrgyz failure.

Then a second mode judges the **real bot's** output (human labels all-pass → reports agreement,
not κ). Because the judge is independently validated, its flags are credible:

```bash
python -m meta.judge_validation live      # → results/judge_validation_live.json
```

It flags the bot failing **~4/7 Kyrgyz vs ~3/15 Russian** — the validated judge localizes the
bot's weakness to Kyrgyz. (Exact counts vary run-to-run; see §8.)

---

## 6. The findings (full detail in `REPORT.md`)

1. **The bot is much weaker on Kyrgyz** — found two independent ways: the validated judge
   (KY agreement ≈0.43 vs RU ≈0.80) and the deterministic language gate (5/7 KY replies given
   in Russian). The bot understands Kyrgyz but answers in Russian.
2. **0 payment leaks** (deterministic gate — the most important check, and it's free/CI-safe).
3. **A grounding bug:** `absent-spa-ru` — asked about a service that is in *neither* hotel
   list, the bot confidently says "we don't have it" instead of deferring. It conflates
   *not-listed* with *known-absent*.

**Found → analysed → fixed → verified.** The Kyrgyz bug was root-caused (a *prompt* bug, not a
model limit — see [`docs/kyrgyz-language-bug.md`](docs/kyrgyz-language-bug.md)) and fixed with
code-side language routing (`sut/hotel_bot/bot_fixed.py`). Re-measured on 500 cases: **language
fidelity 0.74 → 0.99, Kyrgyz 0.76 → 0.92**, Russian and payment safety unharmed
([`reports/suite_report_synth_fixed.md`](reports/suite_report_synth_fixed.md)). Run it yourself:
`python -m evals.run_suite --source synth --limit 500 --variant fixed`.

---

## 7. Golden data

| File | What |
|------|------|
| `data/goldens.jsonl` | 22 hand-labeled scenarios (the test cases): 7 factual, 4 absent-service, 2 off-topic, 3 payment-safety, 2 booking-complete, 2 booking-incomplete, 2 language. All labeled expected-`pass`. Multi-turn cases carry scripted assistant turns. |
| `data/judge_validation_set.jsonl` | 16 *balanced* (pass+fail) fixed replies used ONLY to validate the judge (§5). Separate from the goldens for the variance reason above. |
| `data/system_prompt.txt` | The hotel's data + rules (grounding ground-truth). |

---

## 8. Reproducibility note

The vendored bot calls `gpt-4o-mini` at **default temperature** (faithful to production), so
the SUT is **nondeterministic** — the live eval grid and live-bot agreement numbers shift a
little run-to-run. The findings (KY weakness, 0 payment leaks, the grounding tendency) are
stable; exact counts are representative, not bit-for-bit. The judge-validation **fixture** mode
(§5) does NOT call the SUT — it scores fixed replies — so its κ=1.00 is fully reproducible.

If you ever want deterministic eval numbers, set `temperature=0` in `sut/hotel_bot/bot.py`
(one line) — that's the single deviation from production you'd make.

---

## 9. How to run

```bash
# setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the two keys (see §10)
```

```bash
# OFFLINE — no keys, CI-safe (32 unit tests: metrics, kappa math, loaders, BotRunner via fake client)
pytest tests -q
```

```bash
# LIVE — needs both keys
python -c "from conftest import has_key; print(has_key('OPENAI_API_KEY'), has_key('DEEPSEEK_API_KEY'))"  # expect: True True

pytest evals -v                         # factual / safety / booking / live language gate
python -m meta.judge_validation fixture # the headline: judge κ (RU/KY)
python -m meta.judge_validation live    # validated judge's agreement on the real bot
```

Failures under `pytest evals` are **findings** (the bot failing a criterion), not harness bugs.

**Browsing results:** DeepEval ships `deepeval view` (last run) and `deepeval inspect` (local
TUI) — no signup. For a hosted dashboard with regression-over-time, `deepeval login` syncs to
Confident AI (sends eval data to their servers). This repo's own rollup is `REPORT.md` +
`results/*.json` (gitignored).

### Scaling to many cases + one combined report

The 22 curated goldens are the trusted core. For volume there's a **generated** dataset and a
runner that collapses every case into **one** report — this is "how do I run 1000 cases and
collect the results into one place":

```bash
# 1. Generate 1000 synthetic cases (deterministic templates, 7 kinds x RU/KY) -> data/goldens_synth.jsonl
python -m data.synthesize 1000

# 2. Run the suite and write ONE rollup (results/suite_report_synth.{json,md}).
#    --limit keeps cost down while you iterate; drop it to run all 1000.
python -m evals.run_suite --source synth --limit 60     # a cheap sample
python -m evals.run_suite --source synth                # the full 1000
```

`run_suite` runs each case through the SUT once, applies the deterministic metrics (language,
payment) to every case + the judged Grounding metric to non-booking cases, aggregates with
`meta.aggregate.summarize` (by kind / language / metric + a failures list), and appends a
**cost** line. One file in, one report out.

### Cost — by model (`meta/cost.py` estimator)

Two models are billed: the **SUT** `gpt-4o-mini` (one call per case to produce the reply) and
the **judge** `deepseek-chat` (one call per judged case). The deterministic metrics (payment,
language, slots) cost **$0**.

| Model | Role | $/1M in | $/1M out | ≈ per call |
|-------|------|---------|----------|-----------|
| **gpt-4o-mini** | SUT (bot reply) | $0.15 | $0.60 | ~$0.000153 (≈700 in / 80 out) |
| **deepseek-chat** | judge | $0.27 | $1.10 | ~$0.000309 (≈900 in / 60 out) |

| Volume | gpt-4o-mini (SUT) | deepseek (judge) | **Total** |
|--------|-------------------|------------------|-----------|
| 1 case | $0.00015 | $0.00031 | **$0.00046** |
| **1 000** | **$0.15** | **$0.31** | **≈ $0.46** |
| **10 000** | **$1.53** | **$3.09** | **≈ $4.62** |

So for the model we mostly use here — **`gpt-4o-mini` (the SUT) costs ~$0.15 per 1 000 cases and
~$1.53 per 10 000**; the DeepSeek judge roughly doubles it. If you run only the free deterministic
gates (no judge), 10 000 cases cost just the **$1.53** of gpt-4o-mini calls. Booking cases skip the
grounding judge, so a real mixed run is a bit cheaper than these uniform estimates — the actual
**full 1000-case run cost $0.37** (committed in `reports/suite_report_synth.md`). Every
`run_suite` report prints this run's cost + the projected full-dataset cost, so you decide before spending.

**Full 1000-case run is committed** at [`reports/suite_report_synth.md`](reports/suite_report_synth.md):
1000 cases, 0 errors, ~40 min, **measured cost $0.37**, 2714 metric results, overall pass_rate
**0.855**. It confirms the Kyrgyz weakness at scale — **RU 0.95 vs KY 0.76**, and **83% of all 394
failures are Kyrgyz** — with **0 payment leaks** across all 1000.

> Note: the synthetic cases are graded by the *validated* judge (κ=1.0, §5), not by human
> labels — they measure the bot at volume. The 22 human-curated goldens and the 16-case κ
> fixture remain the trusted, hand-checked core.

### Regression: did a prompt change degrade the bot?

```bash
python -m evals.regression_check --limit 12
```

Runs the same cases under the good prompt and a deliberately weakened one
(`data/system_prompt.regression.txt`, which drops the language + "don't invent" rules), grades
both against the **true** hotel facts, and flags a drop. Demo run: overall 0.93 → 0.90,
grounding −0.10 → `REGRESSION DETECTED`. This is the CI gate for prompt changes.

---

## 10. Keys

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | the **SUT** — runs `gpt-4o-mini` to produce bot replies |
| `DEEPSEEK_API_KEY` | the **judge** — runs `deepseek-chat` to grade replies (out-of-family vs the SUT → no self-preference bias) |

`.env` is gitignored and never committed; only `.env.example` (empty values) is tracked.
Optional overrides in `.env.example`: `SYSTEM_PROMPT_PATH`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_JUDGE_MODEL`.

---

## 11. Repo map (what every file is)

```
sut/                          THE SYSTEM UNDER TEST (vendored real bot)
  hotel_bot/bot.py            faithful copy of hotel-chat-bot/core/bot.py (do not edit logic)
  hotel_bot/db.py             in-memory stand-in for Supabase (reset/set_history helpers)
  bot_runner.py               BotRunner: drives the vendored bot; returns BotOutput
  prompt.py                   load_system_prompt() for judge context

metrics/                      DETERMINISTIC metrics (no key)
  payment_leak.py             scan_payment_leak + PaymentLeakMetric (headline safety gate)
  language_fidelity.py        detect_lang + LanguageFidelityMetric (RU/KY heuristic)
  slot_extraction.py          SlotExtractionMetric (booking slot accuracy)

judge/
  deepseek_judge.py           DeepSeekJudge: wraps DeepSeek as a DeepEvalBaseLLM

golden/
  loader.py                   Golden dataclass + load_goldens() over data/goldens.jsonl

evals/                        LIVE eval tests (skip without keys) — these call the SUT + judge
  test_factual.py             GEval grounding / absent-service / off-topic
  test_safety.py              deterministic payment gate + GEval red-team
  test_booking.py             ConversationalGEval multi-turn booking gate + slot check
  test_language.py            deterministic language-fidelity gate over every live reply
  test_quality.py             AnswerRelevancy (helpful?) + Faithfulness (hallucination?)
  run_suite.py                run a dataset -> ONE aggregated report + cost (the rollup)
  regression_check.py         A/B good vs weakened prompt; flags pass-rate drop

meta/                         "eval of the eval"
  stats.py                    cohens_kappa + confusion_matrix (pure)
  judge_validation.py         κ judge-vs-human, split RU/KY; fixture + live modes (CLI)
  aggregate.py                summarize() many results -> one rollup + to_markdown (pure)
  cost.py                     pricing table + cost estimators (pure)

data/                         system_prompt.txt · system_prompt.regression.txt
                              goldens.jsonl (22 curated) · judge_validation_set.jsonl (16 κ)
                              synthesize.py -> goldens_synth.jsonl (1000 generated)
tests/                        OFFLINE unit tests for everything above (no key, no network) — 92 tests
docs/superpowers/plans/       the implementation plan this repo was built from
REPORT.md                     the results write-up (exact numbers + analysis)
```

---

## 12. Why DeepEval (vs Promptfoo)

A sibling repo evaluates a RAG expert-finder with **Promptfoo** (YAML/CLI, model A/B grid).
DeepEval fits *this* project better because:

- **pytest-native** — evals are Python tests; the live grade becomes a CI gate.
- **Multi-turn** — `ConversationalGEval` / `Turn` grade the full booking dialogue (Promptfoo is weak here).
- **Pluggable judge** — `DeepEvalBaseLLM` lets us inject DeepSeek as an out-of-family judge.

Same rigor (judge validation via κ), different framework — deliberately, to show both.

---

## 13. Tech stack

Python 3.13 · deepeval 4.0.5 · openai 2.41.0 (used for both the gpt-4o-mini SUT and the
OpenAI-compatible DeepSeek endpoint) · pytest 8.2 · python-dotenv. CI (`.github/workflows/`)
runs the offline suite only — no secrets needed.

---

## 14. Limitations / next steps

- Single SUT model (`gpt-4o-mini`); swap and re-run freely.
- The 22 curated goldens + 16 κ-fixture are small **by design** (defensible by hand). The 1000
  synthetic cases give volume but are judge-graded, not human-labeled — so they measure the bot,
  they don't tighten the κ confidence interval (that needs more *human* labels).
- Heuristic language detector (no-ops on very short/`unknown` inputs rather than false-fail).
- No prompt-injection / jailbreak suite yet (high-value next addition to the safety set).
- DeepSeek judge scores via JSON schema (no logprobs) — less calibrated than an OpenAI judge, but keeps the judge out-of-family.
- Cost figures are *estimates* from a pricing table + typical token counts, not metered from API responses.
