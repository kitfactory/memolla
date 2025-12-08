from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


def _hash_vector(text: str, dim: int = 256) -> List[float]:
    # API キーが無い場合のフォールバック埋め込み / Fallback embedding when API key is missing
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(dim):
        b = digest[i % len(digest)]
        vals.append((b - 128) / 128.0)
    return vals


class EmbeddingProvider:
    def __init__(self, *, client: Optional[OpenAI], model: str):
        self.client = client
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            logger.warning("Using hash-based fallback embedding due to missing OpenAI client.")
            return [_hash_vector(t) for t in texts]
        resp = self.client.embeddings.create(model=self.model, input=texts, timeout=30)
        return [item.embedding for item in resp.data]


class LLMProvider:
    def __init__(self, *, client: Optional[OpenAI], model: str):
        self.client = client
        self.model = model

    def summarize(self, text: str) -> str:
        if not self.client:
            # 簡易フォールバック要約 / Simple fallback summary
            return text[:512] + ("..." if len(text) > 512 else "")
        messages = [
            {"role": "system", "content": "Summarize the provided content in concise form."},
            {"role": "user", "content": text},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=30,
            max_tokens=256,
        )
        choice = resp.choices[0].message.content or ""
        return choice.strip()


def build_client(api_key: Optional[str], base_url: Optional[str]) -> Optional[OpenAI]:
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; falling back to local stub providers.")
        return None
    return OpenAI(api_key=api_key, base_url=base_url)
