import streamlit as st
from ingestion import load_pdfs
from vectorstore import create_load_index
from rag_pipeline import run_rag
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.set_page_config(page_title="Production RAG", layout="wide")
st.title("📄 Conversational RAG System")
st.caption("A production-style RAG app with user-isolated vector indexes, conversational memory, and intent-aware responses.")

# Session state
if "db" not in st.session_state:
    st.session_state.db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a very helpful assistant.")
    ]

with st.sidebar:
    files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if files and st.button("Index documents"):
        with st.spinner("Indexing documents..."):
            docs = load_pdfs(files)
            st.session_state.db = create_load_index(docs)
        st.success("Index ready")

query = st.chat_input("Ask a question")

if query:
    if not st.session_state.db:
        st.warning("Please upload and index documents first.")
    else:
        # Store user message
        st.session_state.chat_history.append(
            HumanMessage(content=query)
        )

        with st.spinner("Thinking..."):
            answer, sources = run_rag(
                st.session_state.db,
                query,
                st.session_state.chat_history
            )

        # Store AI response
        st.session_state.chat_history.append(
            AIMessage(content=answer)
        )

        # Render conversation
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant").write(msg.content)
