from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DocumentRecord:
    doc_id: str
    corpus: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    seq: int
    text: str


@dataclass
class MessageRecord:
    session_id: str
    role: str
    raw_content: str
    normalized_content: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    text: str
    score: float
    score_bm25: Optional[float]
    score_dense: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class EvalMetrics:
    recall_at_5: Optional[float]
    mrr_at_10: Optional[float]
    qa_score: Optional[float]
    query_count: int


@dataclass
class TrialConfig:
    params: Dict[str, Any]


@dataclass
class TrialResult:
    config: TrialConfig
    metrics: EvalMetrics


@dataclass
class OptimizeResult:
    level: str
    eval_id: str
    baseline: TrialResult
    best: TrialResult
    trials: List[TrialResult]

    def improved(self) -> bool:
        # baseline と best を簡易比較する / Simple comparison between baseline and best
        return (self.best.metrics.qa_score or 0) > (self.baseline.metrics.qa_score or 0)
