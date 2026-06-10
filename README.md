---
title: Gitanjali v2
emoji: 🕉️
colorFrom: yellow
colorTo: orange
sdk: streamlit
app_file: app.py
pinned: false
---

# 🕉️ Gitanjali v2: The Epic Spiritual Assistant

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-Gitanjali%20v2-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/dkg-2/Gitanjali-v2)
[![GitHub](https://img.shields.io/badge/GitHub-dkg--2%2FGitanjali--v2-181717?style=for-the-badge&logo=github)](https://github.com/dkg-2/Gitanjali-v2)

> *"What if there was someone, like Lord Krishna for Arjun, to talk to me when I felt lost?"*

**Gitanjali v2** is an AI-powered spiritual guide that answers questions from the three pillars of Vedic literature — the **Bhagavad Gita**, the **Mahabharata**, and the **Ramayana** — in your language, with scriptural references.

![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-121212?style=for-the-badge&logo=Chainlink)
![MongoDB](https://img.shields.io/badge/Vector%20Store-MongoDB%20Atlas-47A248?style=for-the-badge&logo=MongoDB)
![Groq](https://img.shields.io/badge/Inference-Groq%20120B-orange?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/Embeddings-BAAI%2FBGE--M3-yellow?style=for-the-badge&logo=huggingface)

---

## 🌟 The Vision

Since I started exploring the **Bhagavad Gita** 🕉️, I wished for a companion 🤝. With 18 chapters and 700 verses, the wisdom of the epics can feel intimidating. I would search for answers, get confused 🤔, and frequently set the book aside.

That's how **Gitanjali** was born. Originally a Gita-only guide, **Version 2** expands into an enlightened companion across all three epics. She speaks in clear language, provides scriptural references, and responds in the language you ask in. It's my way of bringing timeless wisdom into everyday life.

---

## ✨ Key Features

- **Expanded Knowledge Base** — Retrieves wisdom from the Bhagavad Gita, Mahabharata, and Ramayana from a persistent MongoDB Atlas vector store
- **Multilingual Retrieval** — Ask in **English, Hindi, or Sanskrit**. BAAI/BGE-M3 embeddings map all three languages into a shared vector space, enabling cross-lingual retrieval
- **Conversational Memory** — Maintains the last 4 turns of chat history for follow-up questions
- **Scriptural References** — Every answer is grounded in retrieved context with source attribution (book and chapter where available)
- **Real-Time Streaming** — Token-by-token response generation via Groq's LPU inference platform
- **Philosophical Reasoning** — Powered by a 120B-parameter open-source model for nuanced, multi-concept answers

---

## 🏗️ Architecture

Gitanjali v2 uses a **RAG (Retrieval-Augmented Generation)** pipeline with two phases:

### Ingestion (Offline — runs once)
```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → BAAI/BGE-M3 Embeddings → MongoDB Atlas
```

### Inference (Online — runs per query)
```
User Query → BGE-M3 Embed → Atlas Similarity Search (k=3) → LangChain LCEL Chain → Groq 120B → Streamlit
```

### LangChain Chain
```python
chain = prompt | llm | StrOutputParser()
# prompt = ChatPromptTemplate with system context + MessagesPlaceholder + human input
# Streamed via chain.stream({context, input, chat_history})
```

---

## ⚙️ Technical Specifications

| Parameter | Value | Notes |
|---|---|---|
| Embedding Model | `BAAI/bge-m3` | Local CPU inference, no API cost |
| Embedding Dimensions | 1024 | Dense mode, normalized |
| Chunk Size | 1,000 characters | ≈ 180–250 words per chunk |
| Chunk Overlap | 150 characters | 15% overlap to avoid boundary loss |
| Retrieval (k) | 3 documents | Top-3 by cosine similarity |
| Chat History | Last 4 messages | Balances context vs. token cost |
| LLM | `gpt-oss-120b` on Groq | Streaming enabled |
| LLM Temperature | 0.6 | Balanced reasoning |
| Vector Store | MongoDB Atlas | DB: `gitanjali_v2`, Collection: `wisdom_base` |
| Atlas Index | `vector_index` | HNSW algorithm, cosine similarity |
| Frontend | Streamlit | Deployed on HuggingFace Spaces |

---

## 📖 Corpus Details

| Text | Source PDF | Language(s) |
|---|---|---|
| Bhagavad Gita | *Bhagavad-Gita As It Is* | English |
| Mahabharata | *Mahabharata (Unabridged in English)* | English |
| Ramayana | *Ramayana of Valmiki* by Hari Prasad Shastri | English |

> **Total knowledge base:** ~[X] chunks stored in MongoDB Atlas  
> *(Run `db.wisdom_base.countDocuments()` in Atlas to get the exact count)*

---

## 🔄 v1 → v2 Improvements

| Aspect | v1 | v2 |
|---|---|---|
| **Persistence** | Ephemeral Qdrant (in-memory, resets on restart) | MongoDB Atlas (always-on cloud storage) |
| **Corpus** | Bhagavad Gita only | Bhagavad Gita + Mahabharata + Ramayana |
| **Model** | Smaller open-source model | 120B parameter model on Groq |
| **Embeddings** | External API embeddings | Local BAAI/BGE-M3 (zero API cost) |
| **Languages** | English only | English, Hindi, Sanskrit |

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/dkg-2/Gitanjali-v2.git
cd Gitanjali-v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
MONGODB_ATLAS_CLUSTER_URI=your_mongodb_uri
GROQ_API_KEY=your_groq_api_key
DB_NAME=gitanjali_v2
COLLECTION_NAME=wisdom_base
ATLAS_VECTOR_SEARCH_INDEX_NAME=vector_index
MODEL_NAME=gpt-oss-120b
```

### 4. MongoDB Atlas — Create Vector Index
In your Atlas cluster, create a vector search index on the `wisdom_base` collection:
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 1024,
    "similarity": "cosine"
  }]
}
```
Name the index `vector_index`.

### 5. Run Ingestion
If setting up your own database, ingest the PDFs first:
```bash
python ingest.py
```
Or use the provided Google Colab notebook for faster GPU-accelerated ingestion: `Gitanjali_Colab_Ingest.ipynb`

### 6. Launch the App
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Gitanjali-v2/
├── app.py                        # Streamlit app + RAG inference pipeline
├── ingest.py                     # PDF ingestion → chunking → embedding → Atlas
├── Gitanjali_Colab_Ingest.ipynb  # Colab notebook for GPU-accelerated ingestion
├── ARCHITECTURE.md               # Detailed technical breakdown
├── requirements.txt              # Pinned Python dependencies
└── .env                          # Environment variables (not committed)
```

---

## ⚠️ Known Limitations

- **Text-based chunking** — `RecursiveCharacterTextSplitter` uses character boundaries, not verse boundaries. A shloka may be split mid-line. Verse-aware chunking is a planned improvement.
- **k=3 retrieval** — Only the top-3 most similar chunks are passed to the LLM. Questions requiring synthesis across multiple distant passages may receive incomplete answers.
- **No reranking** — Retrieval relies solely on cosine similarity. A cross-encoder reranking step would improve precision.
- **In-session memory only** — Conversation history resets on browser refresh. Persistent cross-session memory is a planned improvement.
- **Verse citations** — Page-level PDF metadata means exact verse numbers (e.g., Gita 2.47) come from the model's training knowledge, not extracted metadata.

---

## 🛣️ Planned Improvements

- [ ] Verse-aware chunking for structured shloka extraction
- [ ] Cross-encoder reranking after initial retrieval
- [ ] RAGAS evaluation pipeline with ground-truth Q&A set
- [ ] Persistent cross-session conversation memory via MongoDB

---

## 🕊️ Mission

Gitanjali's mission is to make the timeless wisdom of the epics accessible, practical, and personal for the modern age.

*"Perform your obligatory duty, for action is superior to inaction." — Bhagavad Gita 3.8*
