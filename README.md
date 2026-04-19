# 🕉️ Gitanjali v2: The Epic Spiritual Assistant

**Gitanjali v2** is an advanced AI-powered spiritual guide designed to provide compassionate wisdom from the three pillars of Vedic literature: the **Bhagavad Gita**, the **Mahabharata**, and the **Ramayana**. 

Built with a focus on high-fidelity reasoning and semantic retrieval, Gitanjali acts not as a search engine, but as an enlightened companion for your spiritual journey.

![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-121212?style=for-the-badge&logo=Chainlink)
![MongoDB](https://img.shields.io/badge/Database-MongoDB%20Atlas-47A248?style=for-the-badge&logo=MongoDB)
![Groq](https://img.shields.io/badge/Inference-Groq%20120B-orange?style=for-the-badge)

---

## 🌟 Key Features

*   **Expanded Knowledge Base**: Seamlessly retrieves wisdom from the Bhagavad Gita, the great epic Mahabharata, and the virtuous journey of the Ramayana.
*   **Deep Reasoning**: Powered by **Groq's GPT-OSS-120B** model, capable of complex philosophical synthesis and empathetic conversation.
*   **Multilingual Mastery**: Speak with Gitanjali in **English, Hindi, or Sanskrit**. She responds in the language of your heart.
*   **Scriptural Integrity**: Every answer is backed by relevant context, providing chapter and verse references whenever possible.
*   **Streaming & Interactive UI**: Real-time response generation with a specialized "Meditation" block that shows the AI's internal reflection.
*   **24/7 Persistence**: Utilizing MongoDB Atlas Vector Search for a high-performance, non-sleeping knowledge base.

---

## 🏗️ Architecture

Gitanjali v2 uses a modern **RAG (Retrieval-Augmented Generation)** pipeline:
1.  **Ingestion**: Scriptures are parsed from PDF, chunked, and embedded using the **BAAI/bge-m3** multilingual model.
2.  **Storage**: Vectors are stored in a cloud-native **MongoDB Atlas Vector Search** index.
3.  **Retrieval**: LangChain performs a similarity search based on the meaning of your question.
4.  **Generation**: The retrieved context and chat history are processed by the 120B parameter model on Groq to produce an enlightened response.

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Gitanjali-v2.git
cd Gitanjali-v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your credentials:
```env
MONGODB_ATLAS_CLUSTER_URI=your_mongodb_uri
GROQ_API_KEY=your_groq_api_key
DB_NAME=gitanjali_v2
COLLECTION_NAME=wisdom_base
ATLAS_VECTOR_SEARCH_INDEX_NAME=vector_index
MODEL_NAME=gpt-oss-120b
```

### 4. Run Ingestion (Optional)
If you are setting up your own database, use the provided script or the Colab notebook:
```bash
python ingest.py
```

### 5. Launch the App
```bash
streamlit run app.py
```

---

## 📚 Project Structure

*   `app.py`: The core Streamlit application and RAG logic.
*   `ingest.py`: Script to process PDFs and upload vectors to MongoDB.
*   `Gitanjali_Colab_Ingest.ipynb`: High-speed GPU ingestion notebook for Google Colab.
*   `ARCHITECTURE.md`: Detailed technical breakdown of the system.
*   `requirements.txt`: List of necessary Python packages.

---

## 🕊️ Mission
Gitanjali's mission is to make the timeless wisdom of the epics accessible, practical, and personal for the modern age. 

*"Perform your obligatory duty, for action is superior to inaction." — Bhagavad Gita 3.8*
