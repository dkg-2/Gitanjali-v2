import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = os.getenv("DB_NAME", "gitanjali_v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wisdom_base")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")

def ingest_data():
    # 1. Initialize MongoDB Client
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    print(f"Clearing existing documents in {COLLECTION_NAME}...")
    collection.delete_many({})

    # 2. Define PDFs to ingest
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    books_dir = os.path.join(base_dir, "books")
    
    pdfs = [
        os.path.join(books_dir, "Bhagavad-Gita As It Is.pdf"),
        os.path.join(books_dir, "Mahabharata (Unabridged in English).pdf"),
        os.path.join(books_dir, "Ramayana.of.Valmiki.by.Hari.Prasad.Shastri.pdf")
    ]

    all_docs = []
    
    # 3. Load and Split PDFs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )

    for pdf_path in pdfs:
        if not os.path.exists(pdf_path):
            print(f"Warning: File not found at {pdf_path}")
            continue
            
        print(f"Loading {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        source_name = os.path.basename(pdf_path).replace(".pdf", "")
        for doc in docs:
            doc.metadata["source"] = source_name
            
        chunks = text_splitter.split_documents(docs)
        print(f"Split {pdf_path} into {len(chunks)} chunks.")
        all_docs.extend(chunks)

    if not all_docs:
        print("❌ Error: No documents found to ingest.")
        return

    # 4. Initialize BGE-M3 Embeddings (Free & Multilingual)
    print("Initializing BGE-M3 Embeddings (this may take a moment on first run)...")
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 5. Upload Documents in Batches
    batch_size = 200 # Smaller batches for local processing
    print(f"Uploading {len(all_docs)} documents to MongoDB Atlas in batches of {batch_size}...")
    
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        if i == 0:
            vector_search = MongoDBAtlasVectorSearch.from_documents(
                documents=batch,
                embedding=embeddings,
                collection=collection,
                index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
            )
        else:
            vector_search.add_documents(documents=batch)
        
        print(f"Processed and Uploaded {min(i + batch_size, len(all_docs))}/{len(all_docs)} documents...")

    print("\n✓ Ingestion complete!")
    print(f"Next Step: In MongoDB Atlas, create a Vector Search Index named '{ATLAS_VECTOR_SEARCH_INDEX_NAME}' with 1024 dimensions.")

if __name__ == "__main__":
    if not MONGODB_ATLAS_CLUSTER_URI:
        print("Error: MONGODB_ATLAS_CLUSTER_URI not found.")
    else:
        ingest_data()
