# memolla – a tiny memory layer for LLM apps

**memolla は、LLM アプリに「記憶」をちょい足しするための薄いメモリ層です。**

- 会話ログとナレッジをまとめて保存  
- **Chroma（Dense ベクトル） + BM25（Lexical）のハイブリッド検索**  
- LLM による簡易要約（OpenAI 互換 API ）

「mem0 みたいなフルスタックまでは要らないけれど、  
**とりあえず長期メモリとそれなりの検索が欲しい**、というときの小さな一枚です。

---

## 特長

- 🧠 **薄いメモリ層**
  - 会話ログ（session/role/content）とナレッジ文書を一緒に保存
  - アプリ側からは `Memory` クラスを触るだけのシンプルな API

- 🔍 **ハイブリッド検索（Chroma + BM25）**
  - キーワードに強い BM25（`bm25s_j`）
  - 意味ベースに強い Dense ベクトル（Chroma）
  - 両者を組み合わせたハイブリッド検索に対応

- 🔄 **モード切り替え**
  - `hybrid`（ハイブリッド）
  - `bm25`（BM25 のみ）
  - `dense`（Chroma のみ）
  - 目的に応じて検索モードを選択可能

- 📝 **簡易要約**
  - OpenAI 互換 API を使って、指定した doc（や会話）をざっくり要約
  - RAG のコンテキストを圧縮する前処理にも利用可能

- 🌱 **小さく始めて、あとから育てられる設計**
  - pip 一発で導入
  - 依存は Chroma + bm25s_j + OpenAI 互換クライアント周辺に限定

---

## インストール

とりあえず試したいだけなら、pip 一発で OK です。

```bash
pip install memolla
```

このリポジトリをクローンして開発したい場合:

```bash
uv sync  # 依存取得

export OPENAI_API_KEY=sk-...  # OpenAI 互換キー
# 必要なら base_url も指定:
#   OPENAI_BASE_URL / LMSTUDIO_BASE_URL / OLLAMA_BASE_URL など
```

※ 実際に使うモデルは OpenAI 互換 API であれば差し替え可能です。

---

## クイックスタート

「会話」「ナレッジ」「検索」「要約」の 4 つだけ覚えれば、だいたい使えます。

```python
from memolla import Memory

mem = Memory()

# --- 会話ログをためる ---
mem.add_conversation("s1", "user", "こんにちは")
mem.add_conversation("s1", "assistant", "こんにちは、memollaです。")

# --- ナレッジを登録する ---
mem.add_knowledge("doc1", "memolla は LLM アプリ向けの薄いメモリ層です。")

# --- ハイブリッド検索（BM25 + Chroma） ---
results = mem.search("メモリ")

# --- ざっくり要約を作る ---
summary = mem.create_summary(doc_id="doc1")

print("search results:", results)
print("summary:", summary)
```

---

## 検索モード

memolla の検索は、Dense（Chroma）と Lexical（BM25）を組み合わせた**ハイブリッド**が基本ですが、  
コンストラクタで **BM25 だけ / Chroma だけ** を固定することもできます。

```python
# コンストラクタでデフォルトモードを指定（デフォルトは hybrid）
mem = Memory(default_mode="hybrid")
mem.search("メモリ")  # default_mode を使用
```

- BM25:  typo に弱いが、「キーワード一致」の強さ・解釈の素直さがメリット  
- Dense:  意味レベルの近さを拾えるが、たまに「それじゃない」ものを連れてくることも  
- Hybrid: 両方のスコアを組み合わせて、実用バランスを狙うモード

### フォールバック動作

Dense 検索（Chroma）が利用できない環境では、**BM25 のみ**の検索に自動フォールバックします。  
このとき、ログに次のような警告を出力します。

- `[mem][W01] ... fallback to bm25`

---

## API 概要

典型的には `Memory` クラスだけを直接触ります。

```python
from memolla import Memory

mem = Memory(
    # ここに必要なら設定を書く（例: DB パスや Chroma パラメータ、デフォルト検索モードなど）
    default_mode="hybrid",  # "bm25" / "dense" も選べます
)
```

### 会話ログ

```python
mem.add_conversation(session_id="s1", role="user", content="こんにちは")

logs = mem.get_conversation("s1")
```

### ナレッジ

```python
mem.add_knowledge(doc_id="doc1", text="memolla の説明文...")
doc = mem.get_knowledge("doc1")
```

### 検索

```python
# ハイブリッド検索
results = mem.search("検索クエリ")  

# モード指定
results_bm25  = mem.search("検索クエリ", mode="bm25")
results_dense = mem.search("検索クエリ", mode="dense")
```

`results` の中身は、スコア・ソース種別（conversation / knowledge）・テキストなどを含む構造体/辞書のリストになる想定です。

### 要約

```python
summary = mem.create_summary(doc_id="doc1")
print(summary)
```

※ 実際の引数や戻り値の形は `memolla/memory.py` の docstring を参照してください。

---

## 想定ユースケース

- 既存の LLM チャットボットに「長期メモリ」を後付けしたい
- 会話ログ + ナレッジを一箇所に集約して、LLM に渡す前の検索・要約だけ自前でコントロールしたい
- mem0 などのリッチなフレームワークの前に、**もっと薄くてシンプルなレイヤー**で実験したい

「DB や RAG 基盤をガチ構築する前に、まずは“動くメモリ付きプロトタイプ”をサクッと作る」  
くらいの位置づけで使ってもらえるとちょうど良いと思います。

---

## 制約 / 現時点の注意点

- Chroma が使えない環境では BM25 のみで検索します（`[mem][W01]` ログを出力）
- 高度なメタデータ管理や複雑な評価ループは対象外です  
  - あくまで「薄いメモリ層」としての役割に絞っています

大きめの設計（メタデータモデル、評価ループ、エージェント統合など）は、  
**上位のアプリや別レイヤーに任せる前提**のライブラリです。

---

## ライセンス / コントリビュート

（ここは実際のライセンスに合わせて調整してください）

- License: MIT / Apache-2.0 など

Issue / PR / アイデア提案など歓迎です。  
「ここにこういうメトリクスが欲しい」「この検索モードも欲しい」などあれば、気軽に投げてください 🙌
