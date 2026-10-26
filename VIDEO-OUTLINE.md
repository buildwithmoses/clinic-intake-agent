# Demo video outline (about 3:00)

**Goal:** show a forward-deployed engineer taking a customer from prototype to production: the problem, the architecture, the human-in-the-loop moment, and the eval gate blocking a bad change.
**Setup before recording:** `.env` filled in, `uv run langgraph dev` running with Studio open, LangSmith project open in a second tab, a terminal ready, a baseline already promoted, and a candidate results file ready.

| Time | On screen | Say (roughly) |
|---|---|---|
| 0:00–0:15 | Face cam, then intake form `intake-002.md` with the chest-pain line highlighted | "A small clinic asked me to automate intake forms. The typing was easy. The hard part is this line: a patient booking a check-up next week who has chest pain on the stairs. Their rule: automate the typing, never automate the judgement." |
| 0:15–0:45 | README mermaid diagram | "A lead agent built on Deep Agents plans and delegates to three specialists: one extracts the form into a typed schema, one answers policy questions with citations, one does triage. Anything with consequences goes to a human." |
| 0:45–1:20 | Studio: send "Please process intake form intake-002." Show the `task` calls fanning out, then the run **pausing** on `escalate_to_clinician`. Click approve. | "The run pauses. That's a LangGraph interrupt. The escalation doesn't happen until a staff member approves, edits or rejects it. Approved cases go into a queue in the LangGraph Store." |
| 1:20–1:45 | `middleware.py` RedFlagGuard, then the `test_guard_fails_closed...` test passing | "I don't trust a prompt to never miss this, so it's enforced in code. The screen writes red flags into graph state. If the model tries to finish without escalating, middleware sends it back, twice, then fails closed." |
| 1:45–2:05 | Studio: "What's the late cancellation fee?" → cited answer. Then the LangSmith trace with `<phone>` / `<email>` masked | "Policy answers come only from the clinic's docs, with file, section and quote. Everything is traced in LangSmith, with patient identifiers masked before they leave the process." |
| 2:05–2:45 | Terminal: `run_evals.py` summary → `gate.py` printing **GATE FAILED** on the candidate. LangSmith experiment comparison view. | "Thirty eval cases cover extraction, policy and urgent-flag recall. Here's a real regression I hit: Haiku got every fact right but silently dropped its citations. Policy accuracy fell from 1.00 to 0.55, overall 0.85 against a baseline of 1.00, so the gate exits non-zero and it doesn't ship. The fix was one line: make the field required. Urgent recall is a hard floor of 100%; it can't average away." |
| 2:45–3:00 | Face cam / `DEPLOYMENT-WRITEUP.md` | "Problem, architecture, what broke, results. That's how I'd run this with a real customer. Code is linked below." |

**B-roll / cutaways:** `uv run pytest` going green (63 tests, no API keys), the `langgraph.json` file, `HOW-IT-WORKS.md` gotchas section (the Claude 5.5 `temperature` / forced-tool-call gotcha is a good 3-second beat).

**Don't:** show `.env`, show real patient data (there is none; keep the "FICTIONAL SAMPLE" banner visible once), or claim numbers that aren't in a LangSmith experiment.
