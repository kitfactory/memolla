# memolla チュートリアル: BM25 / Chroma / Hybrid

本チュートリアルでは、公開 API と内部インデックスを使って検索を試すサンプルを示します。事前に `OPENAI_API_KEY` をセットし、`uv sync` 済みであることを前提とします。

## 1. ハイブリッド検索（デフォルト）
`examples/hybrid_demo.py`  
Memory のデフォルト設定で BM25 + Chroma を併用します。
```bash
python examples/hybrid_demo.py
```
出力例:
```
[hybrid] doc=doc2 chunk=doc2:0 score=0.667 text=bm25 とベクトル検索を合わせたハイブリッド検索を行います。
[hybrid] doc=doc1 chunk=doc1:0 score=0.333 text=memolla は LLM アプリ向けのメモリ層です。
```

## 2. BM25 のみで検索
`examples/bm25_demo.py`  
Dense インデックスを無効化して BM25 のみで検索します。
```bash
python examples/bm25_demo.py
```
出力例:
```
[bm25] doc=doc1 chunk=doc1:0 score=1.000 text=BM25 は単語頻度に基づく検索手法です。
```

## 3. Chroma（ベクトル検索）のみで検索
`examples/chroma_demo.py`  
Chroma の dense インデックスを直接呼び出します（Memory の内部を利用）。
```bash
python examples/chroma_demo.py
```
出力例:
```
[dense] doc=doc1 chunk=doc1:0 score=0.842
[dense] doc=doc2 chunk=doc2:0 score=0.658
```

## メモ
- Dense 利用には埋め込み生成が必要です。`OPENAI_API_KEY` が無い場合は簡易ハッシュ埋め込みで動作しますが、品質は劣化します。
- チャンクサイズ/overlap はデフォルト固定（512/32）。検索パラメータは alpha=0.5, top_k=5, top_k_bm25=10, top_k_dense=10 です。
