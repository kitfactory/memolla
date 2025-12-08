from __future__ import annotations

from typing import List


def chunk_text(text: str, *, chunk_size: int = 512, overlap: int = 32) -> List[str]:
    # シンプルな文字ベース分割 / Simple char-based chunking
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
        if end == len(text):
            break
    if not chunks:
        chunks.append(text)
    return chunks
