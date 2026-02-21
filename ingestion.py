import os, tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

# PDF read function
def load_pdfs(uploaded_files):
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name
        try:
            loader = PyMuPDFLoader(path)
            docs = loader.load()
        finally:
            os.remove(path)
        for d in docs:
            d.metadata["source"] = file.name
        documents.extend(docs)
    return splitter.split_documents(documents)
