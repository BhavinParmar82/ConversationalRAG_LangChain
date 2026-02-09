import os
import shutil
import atexit
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd

import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_pdf_text(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    for file in pdf_path:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=len(text)//10, chunk_overlap=len(text)//100)
    chunks = text_splitters.split_text(text)
    return chunks

def get_faiss_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Also, make sure to provide all the details. 
    If the answer is not available in the provided context then simply say Answer is not available.
    Do not give wrong answer.\n\n
    
    Context: \n {context} \n
    Question:\n {question} \n
    
    Answer : 
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

# Function to clear the FAISS index when the app ends
def clear_faiss_index():
    print("faiss_index is cleared")
    shutil.rmtree('./faiss_index', ignore_errors=True)

# Register the cleanup function
atexit.register(clear_faiss_index)

# Title of the app
st.title('Q&A Genie')

with st.sidebar:
    fileUploader = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    if fileUploader:
        pdf_text = get_pdf_text(fileUploader)
        text_chunks = get_text_chunks(pdf_text)
        faiss_vector_store = get_faiss_vector_store(text_chunks)

# Create a chat input box for the user to ask questions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Handle user input and display the answer
user_question = st.chat_input("Enter your question:")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Get the response from the model
    answer = user_input(user_question)
    st.session_state.messages.append({"role": "assistant", "content": answer["output_text"]})

    # Refresh the chat to show new message
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
