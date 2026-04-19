import streamlit as st
import os
import re
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- Configuration ---
MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = os.getenv("DB_NAME", "gitanjali_v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wisdom_base")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")

# --- Initialize Resources ---
@st.cache_resource
def get_vector_store():
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)

@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0.6, 
        model_name=MODEL_NAME, 
        groq_api_key=GROQ_API_KEY,
        streaming=True
    )

# --- UI Setup ---
st.set_page_config(page_title="Gitanjali v2", page_icon="🕉️", layout="wide")

# Static Premium Theme (Ancient Scroll)
st.markdown("""
<style>
    .stApp {
        background-color: #fdf6e3;
        background-image: url("https://www.transparenttextures.com/patterns/rice-paper-3.png");
        color: #586e75;
    }
    .main-header {
        font-family: 'Georgia', serif; font-size: 3.5rem; font-weight: 900; color: #b58900; text-align: center; margin-top: -50px;
    }
    .sub-header {
        font-family: 'Georgia', serif; font-size: 1.2rem; text-align: center; color: #859900; margin-bottom: 2rem; font-style: italic;
    }
    .think-box {
        background-color: #fcf4dc; padding: 1.2rem; border-radius: 12px; border-left: 6px solid #b58900; font-style: italic; color: #657b83; margin-bottom: 10px;
    }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; margin-right: 5px; background-color: #b58900; color: white;
    }
    section[data-testid="stSidebar"] { background-color: #eee8d5 !important; border-right: 1px solid #93a1a1; }
</style>
""", unsafe_allow_html=True)

# --- Utility ---
def extract_thinking(text):
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text.replace(match.group(0), "").strip()
        return thinking, answer
    return None, text

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'Scripture')}\nContent: {doc.page_content}" for doc in docs)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generating" not in st.session_state:
    st.session_state.generating = False

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🕉️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #b58900;'>Gitanjali v2</h2>", unsafe_allow_html=True)
    
    # STOP BUTTON
    if st.session_state.generating:
        if st.button("⏹️ Stop Generation", use_container_width=True, type="primary"):
            st.session_state.generating = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.markdown('<span class="badge">Bhagavad Gita</span> <br><br><span class="badge">Mahabharata</span> <br><br> <span class="badge">Ramayana</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Guidance")
    if st.button("What is Dharma?", use_container_width=True):
        st.session_state.preset_query = "What is the true meaning of Dharma?"
    if st.button("Anxiety & Peace", use_container_width=True):
        st.session_state.preset_query = "I am feeling anxious. Help me find peace."

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---
st.markdown('<div class="main-header">Gitanjali</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Divine wisdom from the heart of the Epics</div>', unsafe_allow_html=True)

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        thinking, answer = extract_thinking(message["content"])
        if thinking:
            with st.expander("Gitanjali's Meditation"):
                st.markdown(f'<div class="think-box">{thinking}</div>', unsafe_allow_html=True)
        st.markdown(answer)

# Handle Input
query = st.chat_input("Seek guidance...")
if "preset_query" in st.session_state:
    query = st.session_state.preset_query
    del st.session_state.preset_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        st.session_state.generating = True
        
        # Prepare Chain
        llm = get_llm()
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        system_prompt = (
            "Your name is Gitanjali, a divine and enlightened guide specializing in the sacred scriptures: "
            "the Bhagavad Gita, the Mahabharata, and the Ramayana.\n\n"
            
            "PERSONALITY & TONE:\n"
            "You are a warm, compassionate, and conversational companion. "
            "Maintain a spiritual yet practical tone. Explain complex concepts in simple terms.\n\n"
            
            "CORE MANDATES:\n"
            "1. Respond in the language of the question (English, Hindi, or Sanskrit).\n"
            "2. Always provide chapter and verse references for your answers whenever possible.\n"
            "3. If asked for opinions, clearly distinguish between scriptural teachings and personal advice.\n"
            "4. Answer based ON THE PROVIDED CONTEXT. If the context isn't relevant (e.g., greetings), "
            "use your internal enlightened wisdom to maintain the conversation.\n"
            # "5. Always start your response with a brief internal reflection inside <think> tags.\n\n"
            
            "RESPONSE FORMAT:\n"
            # "<think>[Analyze the question and plan your guidance]</think>\n"
            "[Your conversational and enlightened response]\n\n"
            
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Fetch Context
        docs = retriever.invoke(query)
        context = format_docs(docs)
        
        # Prepare History
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
        
        # Chain
        chain = prompt | llm | StrOutputParser()
        
        # Streaming Output
        placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in chain.stream({"context": context, "input": query, "chat_history": history[-4:]}):
                full_response += chunk
                thinking, answer = extract_thinking(full_response)
                
                with placeholder.container():
                    if thinking:
                        with st.expander("Gitanjali's Meditation", expanded=True):
                            st.markdown(f'<div class="think-box">{thinking}</div>', unsafe_allow_html=True)
                    st.markdown(answer + "▌")
            
            thinking, answer = extract_thinking(full_response)
            with placeholder.container():
                if thinking:
                    with st.expander("Gitanjali's Meditation"):
                        st.markdown(f'<div class="think-box">{thinking}</div>', unsafe_allow_html=True)
                st.markdown(answer)
                
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.generating = False
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.generating = False
