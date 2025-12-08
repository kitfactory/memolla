"""Hybrid search demo using Memory default settings."""
from memolla import Memory


def main() -> None:
    mem = Memory()
    mem.add_knowledge("doc1", "memolla は LLM アプリ向けのメモリ層です。")
    mem.add_knowledge("doc2", "bm25 とベクトル検索を合わせたハイブリッド検索を行います。")
    results = mem.search("メモリ層")
    for r in results:
        print(f"[hybrid] doc={r.doc_id} chunk={r.chunk_id} score={r.score:.3f} text={r.text}")


if __name__ == "__main__":
    main()
