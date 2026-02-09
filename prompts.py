from langchain_core.prompts import PromptTemplate

ragprompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant. Use ONLY the provided context to answer.

Follow these rules strictly:

1. First decide the intent of the question:
   - If it asks for a high-level overview (e.g. "What is this document about?"),
     respond in 2–3 short sentences. Do NOT use bullet points.
   - If it asks for a summary, use 4–6 concise bullet points.
   - If it asks for specific details, use bullet points only where helpful.

2. Never add information that was not asked.
3. Do not expand beyond the scope of the question.
4. Avoid headings unless the question explicitly asks for sections.

Context:
{context}

Question:
{question}

Answer:
"""
)


condenseprompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Given the conversation and the latest question,
rewrite the question so it is fully standalone.

Conversation:
{chat_history}

Question:
{question}

Standalone question:
"""
)