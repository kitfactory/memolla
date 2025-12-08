# memolla – a tiny memory layer for LLM apps

**memolla is a thin memory layer to give LLM apps a bit of “long-term memory.”**

- Store conversation logs and knowledge together  
- **Hybrid search with Chroma (dense) + BM25 (lexical)**  
- Lightweight summarization via OpenAI-compatible API

If you don’t need a full stack like mem0 but just want **some long-term memory and decent search**, memolla is the small layer for that.

---

## Highlights

- 🧠 **Thin memory layer**  
  Store conversation logs (session/role/content) and knowledge docs together. The app only touches the `Memory` class.

- 🔍 **Hybrid search (Chroma + BM25)**  
  BM25 (`bm25s_j`) for lexical, Chroma for dense vectors, combined in hybrid mode.

- 🔄 **Mode switching**  
  `hybrid`, `bm25`, `dense` – pick the search mode you want.

- 📝 **Quick summarization**  
  Uses an OpenAI-compatible API to summarize docs (or conversations). Useful for pre-compressing RAG context.

- 🌱 **Start small, grow later**  
  Install via pip; deps are limited to Chroma + bm25s_j + an OpenAI-compatible client.

---

## Install

To try quickly, just install via pip:

```bash
pip install memolla
```

For local dev with this repo:

```bash
uv sync  # install deps

export OPENAI_API_KEY=sk-...  # OpenAI-compatible key
# Optionally set base_url:
#   OPENAI_BASE_URL / LMSTUDIO_BASE_URL / OLLAMA_BASE_URL
```

Any OpenAI-compatible API works; swap the model/base_url as needed.

---

## Quickstart

Remember these four actions: conversations, knowledge, search, summarize.

```python
from memolla import Memory

mem = Memory()

# --- store conversations ---
mem.add_conversation("s1", "user", "Hello")
mem.add_conversation("s1", "assistant", "Hi, I'm memolla.")

# --- register knowledge ---
mem.add_knowledge("doc1", "memolla is a thin memory layer for LLM apps.")

# --- hybrid search (BM25 + Chroma) ---
results = mem.search("memory")

# --- make a quick summary ---
summary = mem.create_summary(doc_id="doc1")

print("search results:", results)
print("summary:", summary)
```

---

## Search modes

Hybrid (Chroma + BM25) is the default, but you can force BM25-only or dense-only:

```python
# default: hybrid
mem.search("memory")

# BM25 only
mem.search("memory", mode="bm25")

# Chroma (dense) only
mem.search("memory", mode="dense")
```

- BM25: strong keyword matching, weaker to typos  
- Dense: semantic closeness, sometimes pulls irrelevant items  
- Hybrid: balances both scores

### Fallback

If Chroma (dense) is unavailable, memolla automatically falls back to BM25-only and logs:

- `[mem][W01] ... fallback to bm25`

---

## API overview

In most cases you only touch `Memory`:

```python
from memolla import Memory

mem = Memory(
    # put settings here if needed (DB path, Chroma params, etc.)
)
```

### Conversations

```python
mem.add_conversation(session_id="s1", role="user", content="Hello")

logs = mem.get_conversation("s1")
```

### Knowledge

```python
mem.add_knowledge(doc_id="doc1", text="Description of memolla...")
doc = mem.get_knowledge("doc1")
```

### Search

```python
# hybrid
results = mem.search("query")

# mode selection
results_bm25  = mem.search("query", mode="bm25")
results_dense = mem.search("query", mode="dense")
```

`results` contain score, source info (conversation/knowledge), and text for each hit.

### Summarize

```python
summary = mem.create_summary(doc_id="doc1")
print(summary)
```

See `memolla/memory.py` for exact signatures/return types.

---

## Use cases

- Add “long-term memory” to an existing LLM chatbot
- Centralize conversation + knowledge, control search/summarize before sending to the LLM
- Experiment with a **thin, simple layer** before adopting a heavier framework (mem0, etc.)

Great for quickly prototyping “LLM + memory” before building a full RAG stack.

---

## Limitations

- If Chroma isn’t available, falls back to BM25-only (logs `[mem][W01]`).
- Advanced metadata management or complex evaluation loops are out of scope; memolla stays a thin memory layer.

Larger design (metadata model, eval loops, agent integrations, etc.) is assumed to live above this library.

---

## License / Contributing

(Adjust to your actual license)

- License: MIT / Apache-2.0, etc.

Issues / PRs / ideas welcome. If you want specific metrics or search modes, please open a ticket 🙌
