import streamlit as st
from ingestion import load_pdfs
from vectorstore import create_load_index
from rag_pipeline import run_rag
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── RAGAS imports ──
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

st.set_page_config(page_title="Production RAG", layout="wide")
st.title("📄 Conversational RAG System")
st.caption("A production-style RAG app with user-isolated vector indexes, conversational memory, and intent-aware responses.")

# ── Session state ──
if "db" not in st.session_state:
    st.session_state.db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a very helpful assistant.")
    ]

# ── NEW: ragas_data without ground_truth ──
if "ragas_data" not in st.session_state:
    st.session_state.ragas_data = {
        "question": [],
        "answer":   [],
        "contexts": [],
    }

# ── Sidebar ──
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

    # ── NEW: RAGAS Evaluation section ──
    st.divider()
    st.subheader("🧪 RAGAS Evaluation")

    # Show how many questions collected so far
    total = len(st.session_state.ragas_data["question"])
    st.caption(f"Questions collected: {total}")

    if st.button("📊 Run Evaluation"):
        if total == 0:
            st.warning("Ask at least one question first!")
        else:
            with st.spinner("Running RAGAS evaluation..."):
                llm        = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                embeddings = OpenAIEmbeddings()

                dataset = Dataset.from_dict(st.session_state.ragas_data)

                results = evaluate(
                    dataset=dataset,
                    metrics=[
                        faithfulness,       # Is answer grounded in context?
                        answer_relevancy,   # Is answer relevant to question?
                    ],
                    llm=llm,
                    embeddings=embeddings,
                )

            st.success("✅ Evaluation Complete!")

            # Show scores as metrics
            df = results.to_pandas()
            st.metric("Faithfulness",      f"{df['faithfulness'].mean():.2f}")
            st.metric("Answer Relevancy",  f"{df['answer_relevancy'].mean():.2f}")

    # ── NEW: Reset evaluation data ──
    if st.button("🔄 Reset Evaluation Data"):
        st.session_state.ragas_data = {
            "question": [],
            "answer":   [],
            "contexts": [],
        }
        st.success("Evaluation data cleared!")

# ── Chat input ──
query = st.chat_input("Ask a question")

if query:
    if not st.session_state.db:
        st.warning("Please upload and index documents first.")
    else:
        st.session_state.chat_history.append(
            HumanMessage(content=query)
        )

        with st.spinner("Thinking..."):
            answer, sources = run_rag(
                st.session_state.db,
                query,
                st.session_state.chat_history
            )

        st.session_state.chat_history.append(
            AIMessage(content=answer)
        )

        # ── NEW: Collect data for RAGAS ──
        contexts = [doc.page_content for doc in sources]

        st.session_state.ragas_data["question"].append(query)
        st.session_state.ragas_data["answer"].append(answer)
        st.session_state.ragas_data["contexts"].append(contexts)

        # ── Render conversation ──
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant").write(msg.content)