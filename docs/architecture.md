# memolla アーキテクチャ (v0)

## 1. 方針
- 公開 API は `Memory` クラスに集約し、内部のストレージ・検索実装は抽象化して差し替え可能にする。
- レイヤー間の依存は単方向（上位→下位）のみとし、ゴッドクラスを避けてシンプルなインターフェースを設計する。
- 原文コーパスを必ず保存し、インデックスは再生成可能なキャッシュとして扱う。

## 2. レイヤー構造と依存方向
- **Interface 層**: `Memory`（アプリから呼ばれる外部 API）。依存先: Application 層。
- **Application 層**: サービス群（ConversationService, KnowledgeService, SearchService, SummaryService, OptimizeService）。依存先: Domain モデル、Infrastructure の抽象 I/F。
- **Domain モデル**: `DocumentRecord` / `ChunkRecord` / `MessageRecord` / `SearchResult` / `OptimizeResult` などのデータ構造。依存先なし。
- **Infrastructure 層**: SQLite 永続化 (`StorageRepository`)、bm25s_j (最新) インデックス (`BM25Index`)、Chroma ベクトルストア (`DenseIndex`)、OpenAI 互換 Embedding/LLM プロバイダ (`EmbeddingProvider` / `LLMProvider`)。依存先: 外部ライブラリ。
- 依存方向: Interface → Application → Domain → Infrastructure。

## 3. データモデル
- `DocumentRecord`: `doc_id`, `corpus`, `metadata`, `created_at`, `updated_at`, `version`。
- `ChunkRecord`: `chunk_id`, `doc_id`, `seq`, `text`。embedding は `DenseIndex` 側に紐付け。
- `MessageRecord`: `session_id`, `role`, `raw_content`, `normalized_content`, `metadata`, `created_at`。
- `SearchResult`: `doc_id`, `chunk_id`, `text`, `score`, `score_bm25`, `score_dense`, `metadata`。
- `OptimizeResult` / `TrialResult` / `EvalMetrics`: 仕様に準拠し、評価結果の比較に用いる。

## 4. 主なインターフェース（例）
```python
# Repository / Index 抽象
class StorageRepository(Protocol):
    def save_message(self, msg: MessageRecord) -> None: ...
    def save_document(self, doc: DocumentRecord, chunks: list[ChunkRecord]) -> None: ...
    def get_session_messages(self, session_id: str) -> list[MessageRecord]: ...
    def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    def list_chunks(self, doc_id: str) -> list[ChunkRecord]: ...

class BM25Index(Protocol):
    def add_chunks(self, chunks: list[ChunkRecord]) -> None: ...
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...  # chunk_id, score

class DenseIndex(Protocol):
    def add_chunks(self, chunks: list[ChunkRecord]) -> None: ...
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...

class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

class LLMProvider(Protocol):
    def generate(self, prompt: str, **options) -> str: ...
```
- `Memory` は上記 I/F 実装をコンポジションし、backend="auto" では SQLiteRepository + bm25s_j (最新) + Chroma + OpenAI 互換の embedding/LLM を束ねる。モデル名が OpenAI 公式なら OpenAI、そうでない場合は base_url などから LMStudio → Ollama の順で判定する。

## 5. フロー概要
- **add_conversation**: Memory → ConversationService → StorageRepository で永続化。
- **add_knowledge**: Memory → KnowledgeService → チャンク生成 → StorageRepository へ保存 → BM25Index/DenseIndex に登録。
- **search**: Memory → SearchService → BM25Index/DenseIndex から取得 → 正規化・スコア融合 → SearchResult を返却。DenseIndex 不在時は BM25 のみ。
- **create_summary**: Memory → SummaryService → データ取得 (messages or corpus) → Summarizer（デフォルト or options で注入）で生成。
- **optimize**: Memory → OptimizeService → 評価ハーネス実行。v0 では `level="eval"` のみ実装し、それ以外は NotImplemented を返す。

## 6. インデックス/検索設計
- BM25: bm25s_j (最新) でチャンク単位のインデックスを構築し、クエリごとにスコアを返す。
- Dense: Chroma (in-process) を用い、OpenAI 互換 EmbeddingProvider で生成したベクトルを登録する。
- ハイブリッド: BM25 と ベクトル（デフォルト Chroma）のスコアを 0〜1 に正規化し、`score = α * vector + (1-α) * bm25` を算出。デフォルトは `alpha=0.5`, `top_k=5`, `fanout=2`（BM25/ベクトルはそれぞれ `top_k * fanout` 件を取得して融合）。
- フォールバック: DenseIndex 初期化失敗時は BM25 のみに切り替え、`[mem][W01]` をログ出力する。

## 7. ログ/エラー方針
- エラーは `[mem][E{番号}]`、警告は `[mem][W{番号}]` で統一する（spec の表 8.1 を参照）。
- 例外は Python の標準例外（ValueError / RuntimeError / NotImplementedError）にメッセージを付与し、上位で捕捉しやすい形にする。
- 重要イベント（インデックス作成、フォールバック発生）は INFO/WARN で記録する。

## 8. 設定と環境変数
- `MEMOLLA_DB_PATH`: SQLite 保存先パス（未設定時はデフォルト位置）。
- `MEMOLLA_CHROMA_PERSIST_DIR`: Chroma の永続化ディレクトリ（未設定時は一時ディレクトリ）。
- `MEMOLLA_EMBEDDING_MODEL`: 埋め込みモデル名（未設定時はデフォルトモデル）。OpenAI 公式名なら OpenAI、その他は base_url などから LMStudio → Ollama を推定。
- `OPENAI_API_KEY`: OpenAI 互換 API 用のキー（優先度は明示指定 → .env → 環境変数）。デフォルトは OpenAI モデル（例: `gpt-4o-mini`, `text-embedding-3-small`）を利用。
- `OPENAI_BASE_URL`: OpenAI 互換 API の Base URL。指定がなければ OpenAI 公式を利用。OpenAI 名で始まらないモデルを指定し、`OPENAI_BASE_URL` が無ければ `LMSTUDIO_BASE_URL` → `OLLAMA_BASE_URL` の順で利用可能なものを選択する。
- `LMSTUDIO_BASE_URL` / `OLLAMA_BASE_URL`: LMStudio / Ollama 向けの OpenAI 互換エンドポイント。`OPENAI_BASE_URL` が無い場合のフォールバックとして使用。
- LLM/Embedding 呼び出し: デフォルトでタイムアウト 30 秒・最大リトライ 2 回・指数バックオフ。`options` や設定で上書き可能にする。
- 必要に応じて shell もしくは `.env`（リポジトリルート想定）で設定する。.env.sample は生成しない。

## 9. 非機能と将来拡張
- pip インストール直後に追加セットアップなしで動作する構成を優先する。
- Backend 実装は DI で差し替え可能にし、pgvector 等の追加も Application 層に影響しないようにする。
- chunk 戦略や embedding モデルは設定で切り替え可能にし、`optimize` で評価比較できるよう拡張する。現状 `optimize` は全 level 未サポート例外。
