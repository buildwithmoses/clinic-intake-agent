# Deployment write-up: Clinic Intake Agent

**Customer:** Larkspur Family Health (fictional family clinic; demo scenario) · **Stack:** Deep Agents + LangGraph + LangSmith + Claude Haiku 5.5 via OpenRouter · **Author:** Henry Moses Jr

> All numbers below come from real LangSmith experiments (linked at the end), run on 2026-10-26.

## Problem

Two front-desk staff retype about 60 intake forms a week and answer the same policy questions from memory. The risk isn't the typing. It's the one line on a form that needs a clinician today ("chest pain on the stairs", "thinking about ending my life", a new amoxicillin script on a penicillin-allergic patient) getting filed as routine. The clinic's requirement: **automate the typing, never automate the judgement.**

Success criteria agreed up front:
1. **Urgent recall = 100%** on the eval set. Hard floor, not averaged in.
2. Extraction field accuracy ≥ 95%. Policy answers always cited, or an honest "not in our policy".
3. Every escalation is approved by a human. Nothing with consequences runs automatically.

## Architecture

- **Lead agent** (`create_deep_agent`, Claude Haiku 5.5 through OpenRouter; one env var swaps in Sonnet 5.5 or any other model) plans and delegates through the `task` tool to three **isolated specialists**, each with one job, its own tools and a Pydantic output schema: `intake-extractor` → `PatientIntake`, `policy-answerer` → `PolicyAnswer` with file/section/quote citations, `triage` → `TriageResult`.
- **Red flags are state, not prose.** `read_intake_form` runs a deterministic, negation-aware screen and writes flags into graph state via `Command`. Flags survive the subagent boundary because of an idempotent reducer.
- **`RedFlagGuardMiddleware`** (custom `after_model` hook): the lead cannot end a turn while any flag lacks an escalation. It nudges twice, then fails closed with an URGENT hand-off to staff.
- **Human-in-the-loop:** `escalate_to_clinician` is in `interrupt_on` (approve / edit / reject). Approved escalations land in the **LangGraph Store** queue.
- **Guardrails in middleware:** transient-only retries with `on_failure="error"`, per-run model/tool/escalation caps, PII redaction (SSN, cards) before the model sees anything, and a LangSmith anonymizer masking phone/email/DOB in traces.
- **Evals:** 30 cases (8 extraction, 10 policy, 12 triage with 6 urgent, including one phrasing the rules deliberately miss), run with LangSmith `evaluate()`. `gate.py` exits 1 if any tracked metric drops below baseline or urgent recall drops below 1.0.

## What broke / tradeoffs

*Found while building (verified against source, before any live run):*
- **`temperature=0` crashes on Claude Sonnet/Opus 5.5.** `langchain-anthropic` raises on non-default sampling params for these models, and forced `tool_choice` isn't supported. The usual "set temperature to 0 for extraction" habit doesn't work here. We rely on structured output (`ProviderStrategy`) and evals for determinism instead.
- **Retry middleware hid an outage as an answer.** `ModelRetryMiddleware` defaults to `on_failure="continue"`: with no API key it retried an *auth* error 4 times, then returned the error text as the agent's reply. In a clinic, that's a confident wrong answer. Fixed with `retry_on=is_transient` (429/5xx/timeouts only) and `on_failure="error"`.
- **Subagent state merge duplicated flags.** Deep Agents passes parent state into a subagent and writes the full result back, so an append reducer double-counted. Fixed with an idempotent union keyed by `patient_ref:category`.
- **`langgraph dev` rejects a custom checkpointer.** The exported graph ships without one (the platform supplies persistence). Scripts and evals inject `InMemorySaver`.
- **Recall over precision, on purpose.** The rule screen flags "chest pain" in a chat message even when the patient is asking hypothetically. That costs a staff member one click to reject. A missed flag can cost a life. Negation handling ("denies chest pain", "Chest pain: No") keeps the false-positive rate tolerable.
- **BM25, not vectors.** Five short policy docs: lexical search gives exact, auditable citations with no extra infra. A unit test checks the right doc is in the top 3 for every policy eval question before the LLM runs.

*From live runs (OpenRouter + Haiku 5.5):*
- **Forced tool calls rejected through OpenRouter.** With a non-Anthropic client class, LangChain falls back to forcing a tool call for structured output, which Claude 5.5 rejects (`tool_choice: type "tool" and "any" are not supported`). Fixed by declaring `ProviderStrategy` (native JSON-schema output) on every specialist.
- **Intermittent `invalid_request` mid-run.** OpenRouter load-balances one model across Anthropic, Bedrock, Vertex and Azure, and not every host honours every parameter. Fixed with provider routing: `require_parameters: true`, Anthropic first.
- **The eval harness had a race.** `evaluate()` runs examples on 4 threads; two threads built two agents with separate checkpointers, so 2 of 12 triage cases read state from the wrong one and crashed. That showed up as urgent recall 0.67, which *looked* like a safety regression but was a harness bug. Fixed with a lock. Lesson: inspect failing cases before trusting a metric.
- **Haiku silently dropped citations.** Policy answers had 100% correct facts but 9 of 9 had *no citations at all*. Root cause: `citations` had a default, so the JSON schema marked it optional and the smaller model omitted the field. Prompt wording didn't fix it (0.65 → 0.55). Making the field required did (→ 1.0). The gate blocks the uncited version.

## Results

| Metric | Baseline (Haiku 5.5, citations required) | Regression (citations optional) | Gate |
|---|---|---|---|
| Overall (30 cases) | **1.00** | 0.85 | **BLOCKED** (exit 1) |
| Extraction field accuracy | 1.00 | 1.00 | ok |
| Policy accuracy (cited + correct) | 1.00 | 0.55 | FAIL |
| Triage accuracy | 1.00 | 1.00 | ok |
| **Urgent recall (floor 1.0)** | **1.00** (6/6) | 1.00 | ok |
| Urgent precision | 1.00 | 1.00 | ok |
| p50 latency | extraction 6.5s · policy 4.2s · triage (full agent) 12.6s | similar | |

Cost: the whole 30-case run used ~437k tokens. At Haiku 5.5's OpenRouter price ($0.10 in / $0.50 out per 1M), the full suite costs cents per run, cheap enough to run on every change. (LangSmith doesn't record OpenRouter cost; exact spend is on the OpenRouter dashboard.)

LangSmith experiments: [baseline (haiku-v3)](https://smith.langchain.com/o/07e7a262-da74-4aa1-9edb-a1bdb7aba595/projects/p/69026019-3ac5-45bb-b820-4b5938949db4) · [regression (haiku-v2)](https://smith.langchain.com/o/07e7a262-da74-4aa1-9edb-a1bdb7aba595/projects/p/9d6b5de6-c4e7-43d1-80e4-7862a8959fb5)

**Next steps for a real rollout:** BAAs and zero-retention settings with each vendor; clinician-owned red-flag rules and a drug database for allergy checks; OCR for scanned PDFs; Postgres checkpointer plus auth on a LangSmith Deployment; a weekly sample of live traces added back into the eval set.
