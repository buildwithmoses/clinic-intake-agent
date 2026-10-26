"""Runtime configuration. Everything is overridable by environment variable."""

from __future__ import annotations

import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Models go through OpenRouter (langchain-openrouter): one OPENROUTER_API_KEY,
# and any provider/model id from openrouter.ai/models. Direct providers still
# work (e.g. anthropic:claude-sonnet-5-5 with ANTHROPIC_API_KEY). Claude
# sonnet/opus 5.5 reject non-default temperature/top_p/top_k and forced
# tool_choice, so we never set those here.
# Haiku 5.5 is ~$0.10/$0.50 per 1M tokens: cheap enough to run the full eval
# suite for cents. Swap to openrouter:anthropic/claude-sonnet-5.5 for max quality.
DEFAULT_MODEL = "openrouter:anthropic/claude-haiku-5.5"

# Shown on OpenRouter's dashboard so spend is attributed to this app.
OPENROUTER_APP_URL = "https://buildwithmoses.com"
OPENROUTER_APP_TITLE = "Clinic Intake Agent"

# OpenRouter load-balances one model across several hosts (Anthropic, Bedrock,
# Vertex, Azure). Specialists rely on native structured output, so only route to
# hosts that honour every parameter we send, and try Anthropic first. Without
# this we saw intermittent invalid_request errors mid-run.
OPENROUTER_PROVIDER = {"require_parameters": True, "order": ["anthropic"], "allow_fallbacks": True}

# Which env var each provider prefix needs, for friendly "missing key" errors.
PROVIDER_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load this project's .env, then fill any unset/blank vars from .env files
    further up (e.g. the buildwithmoses-automations repo's OPENROUTER_API_KEY).
    Real environment variables always win."""
    from dotenv import dotenv_values

    for folder in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        env_file = folder / ".env"
        if env_file.is_file():
            for key, value in dotenv_values(env_file).items():
                if value and not os.environ.get(key):
                    os.environ[key] = value


def data_dir() -> Path:
    return Path(os.getenv("CLINIC_DATA_DIR", PROJECT_ROOT / "data"))


def lead_model_id() -> str:
    return os.getenv("CLINIC_MODEL", DEFAULT_MODEL)


def subagent_model_id() -> str:
    """Specialists can run on a cheaper model (e.g. the free openrouter:nvidia/nemotron-3-super-120b-a12b:free)."""
    return os.getenv("CLINIC_SUBAGENT_MODEL") or lead_model_id()


def get_model(model_id: str) -> BaseChatModel:
    """Build a chat model via init_chat_model.

    Note: ChatOpenRouter checks for OPENROUTER_API_KEY at construction (unlike
    ChatAnthropic, which waits until the first call).
    """
    if model_id.startswith("openrouter:"):
        return init_chat_model(
            model_id,
            app_url=OPENROUTER_APP_URL,
            app_title=OPENROUTER_APP_TITLE,
            openrouter_provider=OPENROUTER_PROVIDER,
        )
    return init_chat_model(model_id)


def missing_model_keys() -> list[str]:
    """Env vars the configured lead/specialist models need but don't have."""
    needed = {
        PROVIDER_KEYS[m.split(":", 1)[0]]
        for m in (lead_model_id(), subagent_model_id())
        if m.split(":", 1)[0] in PROVIDER_KEYS
    }
    return sorted(k for k in needed if not os.getenv(k))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def max_model_calls_per_run() -> int:
    return _int_env("CLINIC_MAX_MODEL_CALLS_PER_RUN", 30)


def max_tool_calls_per_run() -> int:
    return _int_env("CLINIC_MAX_TOOL_CALLS_PER_RUN", 60)


def max_escalations_per_run() -> int:
    return _int_env("CLINIC_MAX_ESCALATIONS_PER_RUN", 10)
