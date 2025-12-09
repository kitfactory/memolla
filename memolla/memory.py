from __future__ import annotations

import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_provider_settings
from .indexes import BM25Index, DenseIndex
from .models import (
    ChunkRecord,
    DocumentRecord,
    MessageRecord,
    OptimizeResult,
    SearchResult,
    TrialConfig,
    TrialResult,
    EvalMetrics,
)
from .providers import EmbeddingProvider, LLMProvider, build_client
from .storage import SQLiteRepository
from .utils import chunk_text

logger = logging.getLogger(__name__)


class Memory:
    def __init__(
        self,
        *,
        backend: str = "auto",
        db_path: str | None = None,
        default_mode: str = "hybrid",
        **backend_options: Any,
    ) -> None:
        if backend != "auto":
            raise ValueError("[mem][E004] Unsupported backend")
        if default_mode not in {"hybrid", "bm25", "chroma"}:
            raise ValueError("[mem][E004] unsupported search mode")
        self.default_mode = default_mode

        self.db_path = Path(db_path or os.getenv("MEMOLLA_DB_PATH", ".memolla/db.sqlite"))
        self.repo = SQLiteRepository(self.db_path)

        provider_settings = load_provider_settings(
            model=backend_options.get("model"),
            embedding_model=backend_options.get("embedding_model"),
            api_key=backend_options.get("api_key"),
            base_url=backend_options.get("base_url"),
        )
        client = build_client(provider_settings.api_key, provider_settings.base_url)
        self.embedding = EmbeddingProvider(client=client, model=provider_settings.embedding_model)
        self.llm = LLMProvider(client=client, model=provider_settings.model)

        chroma_dir = os.getenv("MEMOLLA_CHROMA_PERSIST_DIR")
        self.bm25_index = BM25Index()
        try:
            self.dense_index = DenseIndex(persist_dir=chroma_dir, embedding=self.embedding)
            self.dense_available = True
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("[mem][W01] dense index unavailable, fallback to bm25 (%s)", exc)
            self.dense_available = False
            self.dense_index = None  # type: ignore

    # 会話ログ追加 / add conversation log
    def add_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        if not session_id or not content:
            raise ValueError("[mem][E001] session_id and content are required")
        now = ts or datetime.utcnow()
        msg = MessageRecord(
            session_id=session_id,
            role=role,
            raw_content=content,
            normalized_content=None,
            metadata=metadata or {},
            created_at=now,
        )
        self.repo.save_message(msg)

    # ナレッジ追加 / add knowledge
    def add_knowledge(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.repo.document_exists(doc_id):
            raise ValueError("[mem][E002] doc_id already exists")
        now = datetime.utcnow()
        doc = DocumentRecord(
            doc_id=doc_id,
            corpus=text,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            version=1,
        )
        chunks_raw = chunk_text(text, chunk_size=512, overlap=32)
        chunks: List[ChunkRecord] = []
        for idx, chunk_text_value in enumerate(chunks_raw):
            chunk_id = f"{doc_id}:{idx}"
            chunks.append(ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, seq=idx, text=chunk_text_value))
        self.repo.save_document(doc, chunks)
        self.bm25_index.add_chunks(chunks)
        if self.dense_available and self.dense_index:
            self.dense_index.add_chunks(chunks)

    # 会話取得 / get conversation
    def get_conversation(self, session_id: str) -> List[MessageRecord]:
        return self.repo.get_session_messages(session_id)

    # ナレッジ取得 / get knowledge
    def get_knowledge(self, doc_id: str) -> DocumentRecord:
        doc = self.repo.get_document(doc_id)
        if not doc:
            raise ValueError("[mem][E006] target not found")
        return doc

    # 検索 / search
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if top_k <= 0:
            raise ValueError("[mem][E004] top_k must be positive")
        mode = self.default_mode
        if mode not in {"hybrid", "bm25", "chroma"}:
            raise ValueError("[mem][E004] unsupported search mode")

        bm25_hits: List[tuple[str, float]] = []
        dense_hits: List[tuple[str, float]] = []

        if mode in {"hybrid", "bm25"}:
            bm25_hits = self.bm25_index.search(query, top_k=top_k)

        if mode in {"hybrid", "chroma"}:
            if self.dense_available and self.dense_index:
                try:
                    dense_hits = self.dense_index.search(query, top_k=top_k)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning("[mem][W01] chroma index unavailable, fallback to bm25 (%s)", exc)
                    if mode == "chroma":
                        dense_hits = []
                    else:
                        dense_hits = []
            else:
                logger.warning("[mem][W01] chroma index unavailable, fallback to bm25")
                if mode == "chroma":
                    dense_hits = []

        if mode == "bm25":
            return self._hits_to_results(bm25_hits, top_k, use_bm25=True, use_dense=False)
        if mode == "chroma":
            return self._hits_to_results(dense_hits, top_k, use_bm25=False, use_dense=True)

        merged = self._merge_scores(bm25_hits, dense_hits)
        return self._merged_to_results(merged, top_k)

    def _get_chunk(self, doc_id: str, chunk_id: str) -> Optional[ChunkRecord]:
        chunks = self.repo.list_chunks(doc_id)
        for c in chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def _merge_scores(
        self,
        bm25_hits: List[tuple[str, float]],
        dense_hits: List[tuple[str, float]],
        alpha: float = 0.5,
    ) -> List[tuple[str, float, Optional[float], Optional[float]]]:
        scores: Dict[str, Dict[str, float]] = {}
        if bm25_hits:
            max_bm = max(score for _, score in bm25_hits) or 1.0
            for chunk_id, score in bm25_hits:
                scores.setdefault(chunk_id, {})["bm25"] = score / max_bm
        if dense_hits:
            max_dense = max(score for _, score in dense_hits) or 1.0
            for chunk_id, score in dense_hits:
                scores.setdefault(chunk_id, {})["dense"] = score / max_dense

        merged: List[tuple[str, float, Optional[float], Optional[float]]] = []
        for chunk_id, val in scores.items():
            s_bm = val.get("bm25")
            s_de = val.get("dense")
            if s_bm is None and s_de is None:
                continue
            score = (alpha * (s_de or 0)) + ((1 - alpha) * (s_bm or 0))
            merged.append((chunk_id, score, s_bm, s_de))
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def _hits_to_results(
        self,
        hits: List[tuple[str, float]],
        top_k: int,
        *,
        use_bm25: bool,
        use_dense: bool,
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        if not hits:
            return results
        max_score = max(score for _, score in hits) or 1.0
        for chunk_id, score in hits[:top_k]:
            doc_id = chunk_id.split(":", 1)[0]
            chunk = self._get_chunk(doc_id, chunk_id)
            if not chunk:
                continue
            score_norm = score / max_score
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk.text,
                    score=score_norm,
                    score_bm25=score_norm if use_bm25 else None,
                    score_dense=score_norm if use_dense else None,
                    metadata={},
                )
            )
        return results

    def _merged_to_results(
        self,
        merged: List[tuple[str, float, Optional[float], Optional[float]]],
        top_k: int,
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        for chunk_id, score, sbm25, sdense in merged[:top_k]:
            doc_id = chunk_id.split(":", 1)[0]
            chunk = self._get_chunk(doc_id, chunk_id)
            if not chunk:
                continue
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk.text,
                    score=score,
                    score_bm25=sbm25,
                    score_dense=sdense,
                    metadata={},
                )
            )
        return results

    # 要約 / create_summary
    def create_summary(
        self,
        *,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        **options: Any,
    ) -> str:
        if (session_id and doc_id) or (not session_id and not doc_id):
            raise ValueError("[mem][E003] specify either session_id or doc_id")

        if session_id:
            messages = self.repo.get_session_messages(session_id)
            if not messages:
                raise ValueError("[mem][E006] target not found")
            text = "\n".join(f"{m.role}: {m.raw_content}" for m in messages)
        else:
            doc = self.repo.get_document(doc_id or "")
            if not doc:
                raise ValueError("[mem][E006] target not found")
            text = doc.corpus

        summarizer = options.get("summarizer")
        if summarizer:
            return summarizer(text)
        return self.llm.summarize(text)

    # 最適化 / optimize
    def optimize(
        self,
        *,
        level: str = "eval",
        doc_ids: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        eval_id: Optional[str] = None,
        llm: Any = None,
        dry_run: bool = False,
    ) -> OptimizeResult:
        raise NotImplementedError("[mem][E005] optimize level not implemented")
