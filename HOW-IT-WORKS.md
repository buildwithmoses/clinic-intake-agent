# How It Works: the LangChain stack behind this project

Plain-English notes on each piece, checked against the official docs **and the installed source code** on 2026-10-26. The versions below are what `uv` resolved. Where the docs and the code disagreed, or a detail surprised me, it's marked **Gotcha**.

| Package | Version | What it is |
|---|---|---|
| `deepagents` | 0.7.32 | Agent harness: planning, subagents, virtual filesystem, HITL wiring |
| `langchain` | 1.4.7 | `create_agent`, `init_chat_model`, middleware |
| `langchain-anthropic` | 1.7.6 | Claude chat model (`ChatAnthropic`) |
| `langgraph` | 1.2.17 | The runtime underneath: state, checkpointers, interrupts, Store |
| `langgraph-cli[inmem]` | 0.4.39 | `langgraph dev` local server + Studio |
| `langsmith` | 0.14.12 | Tracing, datasets, `evaluate()` |

---

## 1. LangGraph is the engine

LangGraph runs an agent as a **graph of steps over shared state**. You rarely draw the graph by hand anymore: `create_agent` and `create_deep_agent` build it for you. These are the LangGraph pieces that matter here:

- **State.** A typed dict (here `ClinicState`) that every step reads and writes. Each key can have a **reducer** that decides how updates merge. Our `red_flags` key uses an idempotent union reducer.
- **Checkpointer.** Saves state after every step, keyed by a `thread_id`. This is what lets a run pause and resume later. `InMemorySaver` is fine for scripts and tests; Postgres is for production.
- **Interrupts (human-in-the-loop).** A step can call `interrupt(...)`, which pauses the graph and saves state. You resume later with `Command(resume=...)` on the **same `thread_id`**. "Human-in-the-loop requires checkpointing to handle interrupts."
- **Store.** Long-term memory that lives *across* threads, unlike the checkpointer, which is per thread. You write with `store.put(namespace, key, value)` and read with `store.search(namespace)`. Tools reach it through `runtime.store`. We use it as the clinician escalation queue.

Docs: [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [Persistence (checkpointers + Store)](https://docs.langchain.com/oss/python/langgraph/persistence)

## 2. LangChain v1: `create_agent`, `init_chat_model`, middleware

- **`init_chat_model("anthropic:claude-sonnet-5-5")`** builds a chat model from a `provider:model` string. It's lazy: **no API key is needed until the first call**, which is why our graph compiles in tests with no keys.
- **`create_agent(model, tools, system_prompt, response_format, middleware, state_schema, checkpointer, store)`** is the standard tool-calling agent loop.
- **`response_format`**: pass a Pydantic class and you get `result["structured_response"]`. It uses `ProviderStrategy` (the model's native structured output) when the model's profile supports it, and falls back to `ToolStrategy` (an artificial tool call) otherwise.
- **Middleware** hooks into the loop (`before_agent`, `before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`, ...). This is where **guarantees** belong, because a prompt can be ignored and code can't. Built-ins we use, all from `langchain.agents.middleware`:

| Middleware | What we use it for |
|---|---|
| `ModelRetryMiddleware(max_retries, retry_on, on_failure, backoff_factor, ...)` | Retry only transient model errors (429, 5xx, timeouts) |
| `ToolRetryMiddleware(..., tools=[...])` | Retry read-only tools. **Not** escalations (never double-page a clinician) |
| `ModelCallLimitMiddleware(run_limit, thread_limit, exit_behavior)` | Hard cap on model calls per run |
| `ToolCallLimitMiddleware(tool_name, run_limit, exit_behavior)` | Global tool cap, plus a separate cap on `escalate_to_clinician` |
| `PIIMiddleware(pii_type, strategy, detector, apply_to_input, apply_to_output, apply_to_tool_results)` | Redact SSNs and mask card numbers before the model sees them |
| `HumanInTheLoopMiddleware(interrupt_on=...)` | Installed for us by Deep Agents from `interrupt_on` |

Custom middleware is a subclass of `AgentMiddleware`. A hook can return `{"jump_to": "model" | "end" | "tools"}` if it's decorated with `@hook_config(can_jump_to=[...])`. Our `RedFlagGuardMiddleware` uses this.

**Gotchas**
- `PIIMiddleware` built-in types are only `email`, `credit_card`, `ip`, `mac_address`, `url`. For an SSN you give a custom name plus `detector=` (a regex string or a function).
- `ModelRetryMiddleware` defaults to `on_failure="continue"`. On final failure, the error text comes back **as the AI's answer**. With no API key it retried an auth error 4 times and then "answered" with the error. We set `retry_on=is_transient` and `on_failure="error"`.
- `ToolCallLimitMiddleware` default `exit_behavior` is `"continue"`, but `ModelCallLimitMiddleware` defaults to `"end"`.

Docs: [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) · [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom) · [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) · [Anthropic integration](https://docs.langchain.com/oss/python/integrations/chat/anthropic)

### The Claude model IDs (verified in source)

`langchain-anthropic` 1.7.6 ships model profiles for `claude-sonnet-5-5`, `claude-opus-5-5` and `claude-haiku-5-5` (plus `claude-sonnet-5` / `claude-opus-5`). `init_chat_model("anthropic:claude-sonnet-5-5")` gives a `ChatAnthropic` with `max_tokens=128000`.

**Gotcha (important):** for `claude-sonnet-5-5` and `claude-opus-5-5`, `ChatAnthropic` **raises** if you set a non-default `temperature`, `top_p` or `top_k`. It also **does not support forced `tool_choice`** (`_supports_forced_tool_choice` returns False), and manual `thinking={"type": "enabled", "budget_tokens": ...}` is rejected in favour of adaptive thinking plus `effort`. A lot of copy-pasted LangChain code does `temperature=0`, and that breaks on these models. We set none of those parameters.

## 3. Deep Agents: the harness

`deepagents` wraps `create_agent` with a batteries-included middleware stack:

```python
from deepagents import create_deep_agent
agent = create_deep_agent(
    model=..., tools=[...], system_prompt="...",
    subagents=[{"name": ..., "description": ..., "system_prompt": ..., "tools": [...],
                "model": ..., "middleware": [...], "response_format": PydanticModel}],
    middleware=[...], interrupt_on={"tool_name": {"allowed_decisions": ["approve","edit","reject"]}},
    state_schema=..., checkpointer=..., store=..., name="...",
)
```

What you get for free:
- A **`task` tool** that delegates to subagents. Each subagent runs in an **isolated context**: it sees only the task description, not the whole chat. That keeps the lead's context small. (`mode="fork"` is experimental and inherits the parent's history.)
- A **virtual filesystem** (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), stored in state by default.
- Summarization, prompt caching for Anthropic (automatic), patching of dangling tool calls.
- `interrupt_on` → auto-installs `HumanInTheLoopMiddleware`. Declarative subagents **inherit** the top-level `interrupt_on` unless they set their own.

**Gotchas**
- **Planning (`write_todos`) is now opt-in.** The overview says task planning is "opt-in via `TodoListMiddleware`". Older blog posts show it on by default.
- **A default `general-purpose` subagent is added automatically** unless you name one `general-purpose` or disable it via a `HarnessProfile`. It inherits the lead's tools.
- **Subagent state flows back to the parent.** The `task` tool writes the subagent's final state (minus `messages`, `todos`, `structured_response`, ...) back into the parent. Because the parent's state is also passed *in*, a plain `operator.add` reducer would **duplicate** entries. Our `red_flags` reducer is an idempotent union for exactly this reason.
- `subagent["response_format"]`: the subagent's structured output is JSON-serialized and becomes the `ToolMessage` the lead sees.
- Passing `model=None` (default `claude-sonnet-4-6`) is deprecated. Always pass a model.

Docs: [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview) · [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents) · [Deep Agents HITL](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop) · [Models](https://docs.langchain.com/oss/python/deepagents/models) · [GitHub](https://github.com/langchain-ai/deepagents) · [PyPI](https://pypi.org/project/deepagents/)

### The HITL round trip (exact shapes)

```python
result = agent.invoke({"messages": [...]}, config)          # config has a thread_id
if result.get("__interrupt__"):
    req = result["__interrupt__"][0].value                    # {"action_requests": [...], "review_configs": [...]}
    # one decision per action request, in order:
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    # or {"type": "edit", "edited_action": {"name": "...", "args": {...}}}
    # or {"type": "reject", "message": "why"}
```

A rejected call does **not** execute. The model gets a `ToolMessage` carrying your reason.

## 4. `langgraph-cli` and `langgraph dev`

- Install (project-local, no global tools): `uv add --dev "langgraph-cli[inmem]"`. The `inmem` extra is required for `langgraph dev`.
- `uv run langgraph dev` starts a hot-reloading API server on `http://127.0.0.1:2024` and opens **LangGraph Studio** (hosted at smith.langchain.com, talking to your local server). Useful flags: `--port`, `--no-browser`, `--tunnel` (for Safari / remote), `--allow-blocking`.
- `langgraph.json` tells it what to serve:

```json
{ "dependencies": ["."],
  "graphs": { "clinic_intake": "./src/clinic_intake_agent/graph.py:agent" },
  "env": ".env",
  "python_version": "3.12" }
```

**Gotcha (verified in `langgraph_api/graph.py`):** in local dev, if your exported graph has its own `checkpointer` or `store`, the server **raises** ("Heads up! Your graph ... includes a custom checkpointer"). The platform supplies persistence. So `graph.py` builds the agent *without* them, and scripts/tests pass `InMemorySaver()` explicitly.

Docs: [LangGraph CLI](https://docs.langchain.com/langsmith/cli) · [Run a local server](https://docs.langchain.com/langsmith/local-server) · [Application structure](https://docs.langchain.com/langsmith/application-structure)

## 5. LangSmith: tracing, datasets, `evaluate()`

**Tracing is just env vars.** No code changes are needed for LangChain/LangGraph:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=clinic-intake-agent          # which project the traces land in
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # EU accounts only
LANGSMITH_WORKSPACE_ID=...                     # only if the key spans several workspaces
```

**Redacting PHI from traces.** LangSmith supports `LANGSMITH_HIDE_INPUTS/OUTPUTS=true`, which hides everything. That makes traces useless for debugging. The finer tool is a regex **anonymizer**: `Client(anonymizer=create_anonymizer([{"pattern": re.compile(...), "replace": "<phone>"}]))`. To apply it to LangChain/LangGraph tracing globally, call `langsmith.configure(client=that_client)`. I verified in source that `LangChainTracer.get_client()` returns the globally configured client. Our `tracing.py` masks SSN, phone, email and dates. The model still sees real values; only the uploaded trace is masked. *Not yet verified live: whether the `langgraph dev` server process honours this for runs it starts. Check the first trace in the UI.*

**Evals.**

```python
from langsmith import Client, evaluate
client = Client()
ds = client.create_dataset(dataset_name="clinic-intake-evals-v1")
client.create_examples(dataset_id=ds.id, examples=[{"inputs": {...}, "outputs": {...}, "metadata": {...}}])

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "correctness", "score": ...}

results = evaluate(target_fn, data="clinic-intake-evals-v1", evaluators=[correctness],
                   summary_evaluators=[...], experiment_prefix="baseline",
                   metadata={"model": "..."}, max_concurrency=4)
for row in results:   # row["run"], row["example"], row["evaluation_results"]
    ...
```

`evaluate()` creates an **experiment** in LangSmith that you can compare side by side with earlier ones. Our gate (`evals/gate.py`) is a small piece of CI logic on top. LangSmith stores the scores, and the gate decides whether a change ships.

Docs: [Trace with LangGraph](https://docs.langchain.com/langsmith/trace-with-langgraph) · [Mask inputs/outputs](https://docs.langchain.com/langsmith/mask-inputs-outputs) · [Evaluate an LLM application](https://docs.langchain.com/langsmith/evaluate-llm-application) · [Evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)

## 6. The LangSmith MCP server (so Claude Code can read your traces)

There are **two** official options, and most blog posts only mention the first.

1. **Standalone open-source server**, [langchain-ai/langsmith-mcp-server](https://github.com/langchain-ai/langsmith-mcp-server). You run it locally with `uvx langsmith-mcp-server` and the env var `LANGSMITH_API_KEY`, or use their hosted instance at `https://langsmith-mcp-server.onrender.com/mcp` with a `LANGSMITH-API-KEY` header. The README gives JSON client config only and **no `claude mcp add` line**.
2. **LangSmith Remote MCP**, hosted by LangSmith itself, which "exposes the same tools as the standalone LangSmith MCP Server". The official docs give the exact Claude Code command:

```bash
claude mcp add --transport http -s user langsmith https://api.smith.langchain.com/mcp
# then inside Claude Code: /mcp -> select "langsmith" -> finish OAuth in the browser
# EU accounts: https://eu.api.smith.langchain.com/mcp
```

Auth is OAuth 2.1 with dynamic client registration, so no key is pasted into a config file. Headless clients can send `X-Api-Key` instead. (The standalone server uses a different header, `LANGSMITH-API-KEY`.)

If you prefer the local standalone server, the equivalent Claude Code command is below. I built it from Claude Code's standard `mcp add` syntax plus the README's `uvx` launch; the README doesn't print it:

```bash
claude mcp add langsmith-local -e LANGSMITH_API_KEY=lsv2_... -- uvx langsmith-mcp-server
```

Docs: [LangSmith Remote MCP](https://docs.langchain.com/langsmith/langsmith-remote-mcp) · [Standalone server repo](https://github.com/langchain-ai/langsmith-mcp-server)

## 7. How the pieces fit in this project

```
langgraph dev ──serves──> graph.py:agent  (create_deep_agent, no checkpointer: platform adds one)
                             │
   lead agent (Claude) ──task──> intake-extractor ─ read_intake_form ─> PatientIntake (Pydantic)
        │                 ──task──> policy-answerer ─ search_clinic_policies (BM25) ─> PolicyAnswer + citations
        │                 ──task──> triage ─ read_intake_form / screen_for_red_flags ─> TriageResult
        │
        ├─ red_flags in STATE (written by tools via Command; deterministic rules)
        ├─ RedFlagGuardMiddleware: can't finish while a flag has no escalation -> nudge x2 -> fail closed
        └─ escalate_to_clinician ──interrupt_on──> HUMAN approve/edit/reject ──> LangGraph Store queue

LangSmith: env-var tracing (PHI masked by anonymizer) + evaluate() over 30 cases -> results JSON -> gate.py (exit 1 on regression)
```
