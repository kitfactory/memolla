"""Chroma-only demo by querying dense index directly."""
from memolla import Memory


def main() -> None:
    mem = Memory()
    mem.add_knowledge("doc1", "Chroma はベクトル検索に使われます。")
    mem.add_knowledge("doc2", "埋め込みモデルでテキストをベクトル化します。")

    if not mem.dense_available or mem.dense_index is None:
        print("Dense index unavailable, cannot run Chroma demo.")
        return

    hits = mem.dense_index.search("ベクトル検索", top_k=5)
    for chunk_id, score in hits:
        doc_id = chunk_id.split(":")[0]
        print(f"[dense] doc={doc_id} chunk={chunk_id} score={score:.3f}")


if __name__ == "__main__":
    main()
