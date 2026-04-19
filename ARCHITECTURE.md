# 🕉️ Gitanjali v2: Architectural Overview

Gitanjali v2 is a **Retrieval-Augmented Generation (RAG)** system designed to transform sacred Vedic wisdom into a live, conversational AI assistant.

## 1. High-Level Data Flow

The system operates in two distinct phases: **Ingestion** (building the brain) and **Inference** (answering questions).

### **Knowledge Ingestion (ETL)**
1.  **Load**: PDFs of the Bhagavad Gita, Mahabharata, and Ramayana are parsed using `PyPDFLoader`.
2.  **Chunk**: The text is split into 1,000-character segments with a 150-character overlap.
3.  **Embed**: Each chunk is processed locally by the **BAAI/bge-m3** model to generate a 1,024-dimensional vector.
4.  **Store**: The text, metadata, and vectors are stored in **MongoDB Atlas Vector Search**.

### **Inference Pipeline (The RAG Loop)**
1.  **Query**: The user asks a question via the Streamlit UI.
2.  **Search**: The question is embedded locally, and a "Similarity Search" is performed against MongoDB.
3.  **Augment**: The retrieved segments and the chat history are injected into a specialized System Prompt.
4.  **Generate**: **Groq's GPT-OSS-120B** model processes the prompt and generates a compassionate, scripture-backed response.

---

## 2. Technical Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | LangChain | Manages the RAG logic, memory, and tool connections. |
| **LLM** | Groq (GPT-OSS-120B) | Provides high-scale reasoning and generation. |
| **Vector DB** | MongoDB Atlas | 24/7 persistent cloud storage for vectors and metadata. |
| **Embeddings** | BAAI/bge-m3 | Local, multilingual embedding model (1024D). |
| **Frontend** | Streamlit | Provides the spiritual chat interface. |

---

## 3. Key Improvements in v2

- **Persistence**: Switched from ephemeral Qdrant to 24/7 MongoDB Atlas.
- **Scope**: Expanded to the three primary epics (Gita, Mahabharata, Ramayana).
- **Intelligence**: Upgraded to 120B-parameter reasoning models.
- **Privacy & Cost**: Uses local embeddings (BGE-M3) to avoid external API costs for vector generation.
