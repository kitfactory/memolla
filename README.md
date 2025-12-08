## memolla

LLM アプリ向けの薄いメモリ層。会話ログとナレッジを保存し、BM25 + Chroma のハイブリッド検索と簡易要約を提供します。OpenAI 互換 API を前提にします（デフォルトは OpenAI モデル）。

### セットアップ
```bash
uv sync  # 依存取得
export OPENAI_API_KEY=sk-...  # OpenAI 互換キー
# 必要なら base_url も指定: OPENAI_BASE_URL / LMSTUDIO_BASE_URL / OLLAMA_BASE_URL
```

### 使い方（簡易）
```python
from memolla import Memory

mem = Memory()
mem.add_conversation("s1", "user", "こんにちは")
mem.add_knowledge("doc1", "memolla はメモリ層です")
results = mem.search("メモリ")
summary = mem.create_summary(doc_id="doc1")
```

### 制約
- `optimize` は全 level 未実装で `[mem][E005] optimize level not implemented` を返します。
- Dense 検索が利用できない場合は BM25 のみで検索し、警告 `[mem][W01] ... fallback to bm25` をログ出力します。
