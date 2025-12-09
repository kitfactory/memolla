"""BM25-only demo by bypassing dense index."""
from memolla import Memory


def main() -> None:
    mem = Memory(default_mode="bm25")

    mem.add_knowledge("doc1", "BM25 は単語頻度に基づく検索手法です。")
    mem.add_knowledge("doc2", "ハイブリッド検索では BM25 も利用されます。")

    results = mem.search("BM25 検索")
    for r in results:
        print(f"[bm25] doc={r.doc_id} chunk={r.chunk_id} score={r.score:.3f} text={r.text}")


if __name__ == "__main__":
    main()
