from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


def is_openai_model(model: Optional[str]) -> bool:
    if not model:
        return True
    return model.startswith(("gpt-", "text-embedding-"))


@dataclass
class ProviderSettings:
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    embedding_model: str


def load_provider_settings(
    *,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ProviderSettings:
    """
    OpenAI 互換 API の設定をロードする / Load OpenAI-compatible API settings.
    優先度: 明示指定 -> .env -> 環境変数。
    """
    load_dotenv()
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resolved_embedding = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if base_url:
        resolved_base = base_url
    else:
        # OpenAI 名で始まらない場合は base_url を優先順で解決 / Resolve base_url when not using OpenAI default host
        candidate = os.getenv("OPENAI_BASE_URL")
        if not candidate and not is_openai_model(resolved_model):
            candidate = os.getenv("LMSTUDIO_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
        resolved_base = candidate

    return ProviderSettings(
        api_key=resolved_api_key,
        base_url=resolved_base,
        model=resolved_model,
        embedding_model=resolved_embedding,
    )
