"""Chroma-only demo (vector search)."""
from memolla import Memory


def main() -> None:
    mem = Memory(default_mode="chroma")
    mem.add_knowledge("doc1", "Chroma はベクトル検索に使われます。")
    mem.add_knowledge("doc2", "埋め込みモデルでテキストをベクトル化します。")

    results = mem.search("ベクトル検索")
    for r in results:
        print(f"[chroma] doc={r.doc_id} chunk={r.chunk_id} score={r.score:.3f} text={r.text}")


if __name__ == "__main__":
    main()
