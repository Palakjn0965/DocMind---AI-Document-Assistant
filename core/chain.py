import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
load_dotenv()


def get_llm():
    """
    Returns a Groq LLM instance.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    return llm


def get_custom_prompt():
    template = """
You are a helpful, friendly research assistant who explains things clearly.

STRICT RULES:
1. Answer ONLY from the provided context below — never use outside knowledge
2. If the answer is not in the context, say exactly: "I could not find this in the provided documents"
3. ALWAYS follow the user's formatting or tone instructions — if they ask for simpler language, bullet points, shorter answer, etc. you MUST comply
4. If the user asks a follow-up like "explain simpler" or "summarize that" — rephrase your previous answer accordingly using the chat history
5. Always mention the page number when referencing specific information

CONTEXT FROM DOCUMENTS:
{context}

CHAT HISTORY:
{chat_history}

USER'S QUESTION OR INSTRUCTION: {question}

YOUR RESPONSE:
"""
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )
    return prompt


def create_qa_chain(retriever):
    """
    Creates a ConversationalRetrievalChain.
    Takes the retriever from Phase 3.
    Returns a chain that can answer questions with memory.
    """
    llm = get_llm()

    # Memory keeps track of previous questions and answers
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # returns which chunks were used
        combine_docs_chain_kwargs={"prompt": get_custom_prompt()}
    )
    return qa_chain


def ask_question(qa_chain, question: str):
    """
    Takes the chain and a question string.
    Returns the answer + source documents.
    """
    response = qa_chain.invoke({"question": question})

    answer = response["answer"]
    source_docs = response["source_documents"]

    return answer, source_docs