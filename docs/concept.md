# memolla コンセプト

## 1. プロダクト概要
- LLM アプリ向けに「会話ログとナレッジを一元管理する薄いメモリ層」を提供し、`add` / `search` / `summarize` / `optimize` の最小 API で運用できることを目指す。
- pip インストール直後に SQLite + bm25s_j + Chroma (in-process) で動作し、外部ミドルウェアを不要にする。
- 原文コーパスを必ず保持し、後続の re-chunk / re-embed / 再要約ができるようにする。

## 2. 想定ユーザーと困りごと
- 対象: LLM アプリ開発者（個人〜小規模チーム）。
- 困りごと: 会話ログ・ドキュメント検索・要約・評価を簡易に扱いたいが、バックエンド選定やインデックス運用に手間をかけたくない。将来の最適化に備えたい。

## 3. 機能一覧（Spec ID 付き）

| Spec ID | 機能 | 説明 | 依存 | フェーズ |
| --- | --- | --- | --- | --- |
| F-00 | Memory 初期化とバックエンド自動選択 | `backend="auto"` 時に SQLite + bm25s_j + Chroma を準備し、パス等はデフォルトで決定。 | bm25s_j, chromadb, SQLite | MVP(v0) |
| F-01 | 会話ログ追加 (`add_conversation`) | セッション単位で role/content/metadata/timestamp を保存。 | F-00 (永続化) | MVP(v0) |
| F-02 | ナレッジ追加 (`add_knowledge`) | doc_id 単位で原文コーパスを保存し、チャンク分割（v0 は `chunk_size=512`, `overlap=32` 固定）して BM25 / ベクトルへ登録。 | F-00 (永続化), F-06 (embedding) | MVP(v0) |
| F-03 | 検索 (`search`) | BM25 + Chroma のハイブリッド検索と簡易 rerank で上位を返す。Chroma 不在時は BM25 単独。α/TopK の初期値は `alpha=0.5`, `top_k=5`, `top_k_bm25=10`, `top_k_dense=10`。 | F-02 (インデックス) | MVP(v0) |
| F-04 | 要約 (`create_summary`) | session または doc を要約する。デフォルトサマライザは OpenAI 互換 LLM（タイムアウト 30s, リトライ 2 回, 指数バックオフ）。利用者提供のコールバックも可。 | F-01 or F-02 (データ源) | MVP(v0) |
| F-05 | 最適化 (`optimize`) | シグネチャと戻り値構造を固定し、v0 は全 level 未サポート例外で返す。 | F-02, F-03 | MVP(v0)〜拡張(v1+) |
| F-06 | Embedding/LLM プロバイダ | OpenAI 互換 API を利用し、モデル名が OpenAI 名なら OpenAI を使用。そうでなければ `OPENAI_BASE_URL` を確認し、無ければ `LMSTUDIO_BASE_URL` → `OLLAMA_BASE_URL` の順で判定。デフォルトは OpenAI モデル（例: `gpt-4o-mini` / `text-embedding-3-small`）を使用し、`OPENAI_API_KEY` が必要。 |  | MVP(v0)〜拡張 |

## 4. フェーズ分け
- MVP (v0): F-00〜F-06 の基本挙動を実装し、`optimize` は全 level 未サポート例外で返す。チャンク戦略 (`chunk_size=512`, `overlap=32`) と検索パラメータ (`alpha=0.5`, `top_k=5`, `top_k_bm25=10`, `top_k_dense=10`) は固定。デフォルトの LLM/Embedding は OpenAI（`OPENAI_API_KEY` 必須）を利用。
- 拡張 (v1+): `optimize(level="index"|"reembed")` のフル試行、複数 embedding モデルや chunk 戦略・検索パラメータの設定化、評価コーパス生成と比較の自動化。

## 5. 依存関係の概要
- 検索と最適化はナレッジのチャンク・インデックス（F-02）に依存。
- 要約は会話ログまたはドキュメント保存（F-01/F-02）に依存し、OpenAI 互換 LLM/embedding は抽象インターフェース（F-06）経由で利用。
- バックエンド差し替え時も公開 API は固定し、内部実装のみ置換可能とする。

## 6. 合意事項
- 公開メソッドと `OptimizeResult` 構造は後方互換を優先し、将来の最適化機能追加時もシグネチャは維持する。
- 原文コーパスは必ず保存し、再処理のために破棄しない。
- LLM/Embedding は OpenAI 互換 API 前提とし、モデル名で OpenAI を判定し、その他は base_url から LMStudio→Ollama の順に推定する。設定は「明示指定 → .env → 環境変数」の優先順とする。
