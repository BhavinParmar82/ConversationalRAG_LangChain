# rag_pipeline.py
from langchain_openai import ChatOpenAI
from config import LLM_MODEL, TOP_K
from prompts import ragprompt, condenseprompt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

import os
if os.getenv("K_SERVICE") is None:
    load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

def run_rag(db, query, chat_history):
    # 1. Convert chat history to plain text
    history_text = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            history_text += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    # 2. Condense question
    standalone_question = llm.invoke(condenseprompt.format(chat_history=history_text, question=query)).content

    # 3. Retrieve
    docs = db.similarity_search(standalone_question, k=TOP_K)

    context = "\n\n".join(doc.page_content for doc in docs)

    # 4. Answer
    prompt = ragprompt.format(context=context, question=standalone_question)

    response = llm.invoke(prompt)

    return response.content, docs
