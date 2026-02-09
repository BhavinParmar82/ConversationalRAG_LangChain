import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from config import EMBEDDING_MODEL
import shutil
import uuid

def get_embeddings():
    return OpenAIEmbeddings(model = EMBEDDING_MODEL)

def create_load_index(docs = None):
    embeddings = get_embeddings()
    INDEX_PATH = f"faiss_index_{uuid.uuid4().hex}"
    
    # 🔥 delete existing index
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_PATH)
    return db