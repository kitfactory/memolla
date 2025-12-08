from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import ChunkRecord, DocumentRecord, MessageRecord


class SQLiteRepository:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                corpus TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                normalized_content TEXT,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def save_document(self, doc: DocumentRecord, chunks: List[ChunkRecord]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO documents (doc_id, corpus, metadata, created_at, updated_at, version) VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc.doc_id,
                doc.corpus,
                json.dumps(doc.metadata),
                doc.created_at.isoformat(),
                doc.updated_at.isoformat(),
                doc.version,
            ),
        )
        cur.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, seq, text) VALUES (?, ?, ?, ?)",
            [(c.chunk_id, c.doc_id, c.seq, c.text) for c in chunks],
        )
        self.conn.commit()

    def document_exists(self, doc_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.fetchone() is not None

    def save_message(self, msg: MessageRecord) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (session_id, role, raw_content, normalized_content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                msg.session_id,
                msg.role,
                msg.raw_content,
                msg.normalized_content,
                json.dumps(msg.metadata),
                msg.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_session_messages(self, session_id: str) -> List[MessageRecord]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT session_id, role, raw_content, normalized_content, metadata, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = cur.fetchall()
        return [
            MessageRecord(
                session_id=row["session_id"],
                role=row["role"],
                raw_content=row["raw_content"],
                normalized_content=row["normalized_content"],
                metadata=json.loads(row["metadata"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT doc_id, corpus, metadata, created_at, updated_at, version FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return DocumentRecord(
            doc_id=row["doc_id"],
            corpus=row["corpus"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            version=row["version"],
        )

    def list_chunks(self, doc_id: str) -> List[ChunkRecord]:
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id, doc_id, seq, text FROM chunks WHERE doc_id = ? ORDER BY seq ASC", (doc_id,))
        rows = cur.fetchall()
        return [
            ChunkRecord(chunk_id=row["chunk_id"], doc_id=row["doc_id"], seq=row["seq"], text=row["text"])
            for row in rows
        ]
