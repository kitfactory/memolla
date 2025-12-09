from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from bm25s import BM25, Tokenizer
from chromadb.config import Settings

from .models import ChunkRecord
from .providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, *, base_dir: Optional[Path] = None) -> None:
        self.tokenizer = Tokenizer()
        self.bm25 = BM25()
        self._encoded_corpus: List[str] = []
        self._chunk_map: Dict[str, ChunkRecord] = {}
        self.base_dir = base_dir
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._load()

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
        self._save()

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

    def _save(self) -> None:
        if not self.base_dir:
            return
        data_path = self.base_dir / "bm25_corpus.json"
        map_path = self.base_dir / "bm25_chunks.json"
        model_dir = self.base_dir / "bm25_model"
        try:
            data_path.write_text(json.dumps(self._encoded_corpus, ensure_ascii=False))
            map_payload = {cid: {"doc_id": c.doc_id, "seq": c.seq, "text": c.text} for cid, c in self._chunk_map.items()}
            map_path.write_text(json.dumps(map_payload, ensure_ascii=False))
            model_dir.mkdir(parents=True, exist_ok=True)
            self.bm25.save(model_dir)
        except Exception:
            logger.exception("Failed to save BM25 index")

    def _load(self) -> None:
        data_path = self.base_dir / "bm25_corpus.json"
        map_path = self.base_dir / "bm25_chunks.json"
        model_dir = self.base_dir / "bm25_model"
        if not data_path.exists() or not map_path.exists() or not model_dir.exists():
            return
        try:
            self._encoded_corpus = json.loads(data_path.read_text())
            raw_map = json.loads(map_path.read_text())
            for cid, val in raw_map.items():
                self._chunk_map[cid] = ChunkRecord(chunk_id=cid, doc_id=val["doc_id"], seq=val["seq"], text=val["text"])
            self.bm25.load(model_dir)
        except Exception:
            logger.exception("Failed to load BM25 index; falling back to empty index")
            self._encoded_corpus = []
            self._chunk_map = {}


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
