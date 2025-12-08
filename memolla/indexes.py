from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import chromadb
from bm25s import BM25, Tokenizer
from chromadb.config import Settings

from .models import ChunkRecord
from .providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self) -> None:
        self.tokenizer = Tokenizer()
        self.bm25 = BM25()
        self._encoded_corpus: List[str] = []
        self._chunk_map: Dict[str, ChunkRecord] = {}

    def _encode_entry(self, chunk: ChunkRecord) -> str:
        return f"{chunk.chunk_id}||{chunk.text}"

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            self._chunk_map[chunk.chunk_id] = chunk
            self._encoded_corpus.append(self._encode_entry(chunk))

        tokenized = self.tokenizer.tokenize(self._encoded_corpus, return_as="tuple", show_progress=False)
        # チャンク全体で再構築する / Rebuild index over full corpus
        self.bm25.index(tokenized, show_progress=False)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self._encoded_corpus:
            return []
        tokenized_query = self.tokenizer.tokenize([query], return_as="tuple", update_vocab=False, show_progress=False)
        res = self.bm25.retrieve(
            tokenized_query,
            corpus=self._encoded_corpus,
            k=top_k,
            return_as="tuple",
            show_progress=False,
            leave_progress=False,
        )
        documents = res.documents[0]
        scores = res.scores[0]
        hits: List[Tuple[str, float]] = []
        for doc, score in zip(documents, scores):
            chunk_id, text = doc.split("||", 1)
            stored = self._chunk_map.get(chunk_id)
            if stored is None:
                continue
            hits.append((chunk_id, float(score)))
        return hits


class DenseIndex:
    def __init__(self, *, persist_dir: Optional[str], embedding: EmbeddingProvider):
        self.embedding = embedding
        if persist_dir:
            settings = Settings(is_persistent=True, persist_directory=persist_dir, anonymized_telemetry=False)
        else:
            settings = Settings(anonymized_telemetry=False)
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection(name="memolla_chunks")
        self._chunk_cache: Dict[str, ChunkRecord] = {}

    def add_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedding.embed_texts(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [{"doc_id": c.doc_id, "seq": c.seq} for c in chunks]
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
        for chunk in chunks:
            self._chunk_cache[chunk.chunk_id] = chunk

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self._chunk_cache:
            return []
        query_emb = self.embedding.embed_texts([query])[0]
        res = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        ids = res.get("ids", [[]])[0]
        scores = res.get("distances") or res.get("embeddings") or [[]]
        score_list: List[float] = []
        if scores:
            # Chroma returns distance; convert to similarity / 類似度に変換
            for d in scores[0]:
                score_list.append(float(1 / (1 + d)))
        hits: List[Tuple[str, float]] = []
        for chunk_id, score in zip(ids, score_list):
            if chunk_id in self._chunk_cache:
                hits.append((chunk_id, score))
        return hits
