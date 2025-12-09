# memolla 仕様 (v0)

## 1. 共通事項
- 対象 API: `memolla.Memory` クラスの公開メソッド（F-00〜F-05）。
- エラーメッセージ形式: `[mem][E{番号}] {説明}`。実装は表 8.1 のメッセージと完全一致させる。
- タイムスタンプは UTC で保存する。
- LLM/Embedding は OpenAI 互換 API を利用し、設定の優先度は「明示指定 → .env → 環境変数」とする。モデル名が OpenAI 公式名なら OpenAI プロバイダ、それ以外は base_url により `OPENAI_BASE_URL` → `LMSTUDIO_BASE_URL` → `OLLAMA_BASE_URL` の順で自動判定する。デフォルトは OpenAI モデル（例: `gpt-4o-mini`, `text-embedding-3-small`）を使用し、`OPENAI_API_KEY` が必要。
- デフォルトの検索パラメータは `alpha=0.5`, `top_k=5`, `fanout=2`。チャンク戦略は `chunk_size=512`, `overlap=32` で固定。
- LLM/Embedding 呼び出しはデフォルトでタイムアウト 30 秒・最大リトライ 2 回・指数バックオフを適用し、呼び出しオプションで上書き可能とする。

## 2. Memory 初期化（Spec ID: F-00）

### 2.1. backend="auto" で初期化する場合、デフォルトストレージを準備する（F-00-01）
- Given `backend="auto"` かつ `db_path` 未指定
- When `Memory()` を生成する
- Then デフォルトの SQLite パスを作成し、bm25s_j (最新) と Chroma (in-process) のインデックスを初期化する
- And `backend_options` があれば bm25/chroma のパラメータに委譲する
- And 環境変数・.env・明示指定から OpenAI 互換 API 設定をロードし、Embedding/LLM プロバイダを初期化する

### 2.2. 未対応 backend を指定した場合は例外を送出する（F-00-02）
- Given `backend` に未サポート値を指定
- When `Memory()` を生成する
- Then `[mem][E004] Unsupported backend` を含む例外を送出する

## 3. 会話ログ追加 add_conversation（Spec ID: F-01）

### 3.1. 正常入力の場合、メッセージを時系列で保存する（F-01-01）
- Given `session_id`・`role`・`content` が非空文字列
- When `add_conversation` を呼ぶ
- Then `MessageRecord` を `created_at` 昇順で保持し、`metadata` は指定がなければ空オブジェクトとして保存する
- And `ts` 未指定の場合は現在時刻 (UTC) を自動採用する

### 3.2. 入力が欠落している場合は検証エラーを返す（F-01-02）
- Given `session_id` または `content` が空
- When `add_conversation` を呼ぶ
- Then `[mem][E001] session_id and content are required` の例外を送出し、永続化しない

## 4. ナレッジ追加 add_knowledge（Spec ID: F-02）

### 4.1. 新規 doc_id で原文コーパスを保存する（F-02-01）
- Given `doc_id` と `text` が非空
- When `add_knowledge` を呼ぶ
- Then `DocumentRecord` を作成し、`corpus` に text を丸ごと保存し、`version=1` を設定する

### 4.2. チャンク分割とインデックスを構築する（F-02-02）
- Given 正常な `add_knowledge` 呼び出し
- When 文書を保存する
- Then 固定のチャンク戦略（文字ベース `chunk_size=512`, `overlap=32`）で `ChunkRecord` を生成する
- And 各チャンクを BM25 と Chroma に登録し、登録結果をストアにコミットする
- And Embedding は OpenAI 互換 API で生成し、プロバイダはモデル名または base_url により OpenAI / LMStudio / Ollama の順で判定する

### 4.3. doc_id が重複した場合は保存を拒否する（F-02-03）
- Given 既存の `doc_id` で `add_knowledge` を呼ぶ
- When 処理する
- Then `[mem][E002] doc_id already exists` の例外を送出し、既存データを変更しない

## 5. 検索 search（Spec ID: F-03）

### 5.1. ハイブリッド検索で結果を統合する（F-03-01）
- Given BM25 とベクトル検索（デフォルト Chroma）が利用可能
- When `search(query, top_k)` を呼ぶ
- Then BM25 から `top_k * fanout` 件、ベクトルから `top_k * fanout` 件を取得し、`score = α * vector + (1-α) * bm25` で正規化スコアを計算する
- And `score` 降順で `top_k` 件の `SearchResult` を返す（`score_bm25`・`score_dense` を含む）
- And 初期値は `alpha=0.5`, `top_k=5`, `fanout=2`

### 5.2. Chroma が利用できない場合は BM25 のみで検索する（F-03-02）
- Given Chroma 初期化に失敗している、または無効化されている
- When `search` を呼ぶ
- Then BM25 の結果のみを `score` として返し、ログに `[mem][W01] chroma index unavailable, fallback to bm25` を記録する

### 5.3. 無効な top_k はエラーにする（F-03-03）
- Given `top_k <= 0`
- When `search` を呼ぶ
- Then `[mem][E004] top_k must be positive` の例外を送出する

## 6. 要約 create_summary（Spec ID: F-04）

### 6.1. session_id を指定した場合は会話ログを要約する（F-04-01）
- Given `session_id` が指定され `doc_id` は None
- When `create_summary(session_id=...)` を呼ぶ
- Then `session_id` のメッセージを時系列で連結し、デフォルトサマライザ（OpenAI 互換 LLM; もしくは `options["summarizer"]`）で要約文字列を返す
- And LLM 呼び出しはデフォルトでタイムアウト 30 秒・最大リトライ 2 回・指数バックオフを適用する（オプションで上書き可）

### 6.2. doc_id を指定した場合はナレッジを要約する（F-04-02）
- Given `doc_id` が指定され `session_id` は None
- When `create_summary(doc_id=...)` を呼ぶ
- Then `DocumentRecord.corpus` を LLM サマライザで要約し、文字列を返す
- And LLM 呼び出しはデフォルトでタイムアウト 30 秒・最大リトライ 2 回・指数バックオフを適用する（オプションで上書き可）

### 6.3. 入力が無効な場合はエラーを返す（F-04-03）
- Given `session_id` と `doc_id` を同時指定、または両方 None
- When `create_summary` を呼ぶ
- Then `[mem][E003] specify either session_id or doc_id` の例外を送出する

### 6.4. 対象が存在しない場合はエラーを返す（F-04-04）
- Given 指定した `session_id` または `doc_id` が存在しない
- When `create_summary` を呼ぶ
- Then `[mem][E006] target not found` の例外を送出する

## 7. 最適化 optimize（Spec ID: F-05）

### 7.1. すべての level は未サポート例外を返す（F-05-01）
- Given `optimize` を任意の `level` で呼ぶ
- When 実行する
- Then `[mem][E005] optimize level not implemented` の NotImplementedError を送出する

## 8. エラー/メッセージ一覧

| ID | メッセージ | 使用箇所 |
| --- | --- | --- |
| [mem][E001] session_id and content are required | add_conversation 入力不足 |
| [mem][E002] doc_id already exists | add_knowledge 重複 |
| [mem][E003] specify either session_id or doc_id | create_summary 入力排他 |
| [mem][E004] top_k must be positive / Unsupported backend | search パラメータ検証 / Memory backend 検証 |
| [mem][E005] optimize level not implemented | optimize すべての level |
| [mem][E006] target not found | create_summary 対象不在 |
| [mem][W01] dense index unavailable, fallback to bm25 | search でベクトル索引不在時の警告ログ |
