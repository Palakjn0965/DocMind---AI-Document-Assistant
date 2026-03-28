import os
import streamlit as st
from core.loader import load_document, split_documents
from core.embedder import create_vector_store, save_vector_store, load_vector_store, get_retriever
from core.chain import create_qa_chain, ask_question

st.set_page_config(
    page_title="DocMind",
    page_icon="📚",
    layout="wide"
)

# title
st.title("📚 DocMind - Your AI Document Assistant")
st.caption("Upload PDFs and chat with your documents")

def init_session_state():
    """
    Streamlit reruns the entire script on every interaction.
    Session state persists variables across reruns.
    """
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []


init_session_state()


# for sidebar
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button("Process Documents", type="primary")

    if process_button and uploaded_files:
        with st.spinner("Processing your documents..."):
            # Saves uploaded files to data folder
            all_chunks = []
            os.makedirs("data", exist_ok=True)

            for uploaded_file in uploaded_files:
                # Save file to disk
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load and chunk it
                pages = load_document(file_path)
                chunks = split_documents(pages)
                all_chunks.extend(chunks)

            # Create fresh vector store from all uploaded docs
            if os.path.exists("vector_store"):
                import shutil
                shutil.rmtree("vector_store")

            vector_store = create_vector_store(all_chunks)
            save_vector_store(vector_store, "vector_store")

            # Build QA chain
            retriever = get_retriever(vector_store, k=4)
            st.session_state.qa_chain = create_qa_chain(retriever)
            st.session_state.chat_history = []
            st.session_state.uploaded_files = [f.name for f in uploaded_files]

        st.success(f"✅ {len(uploaded_files)} document(s) processed!")

    # Show list of uploaded files
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📄 Loaded Documents")
        for fname in st.session_state.uploaded_files:
            st.write(f"• {fname}")

        if st.button("🗑️ Clear All"):
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            if os.path.exists("vector_store"):
                import shutil
                shutil.rmtree("vector_store")
            st.rerun()


# ---- Main Chat Area ----
if st.session_state.qa_chain is None:
    # No documents loaded yet
    st.info("👈 Upload PDFs from the sidebar and click 'Process Documents' to get started")

else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("📎 Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source}")

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)

        # Add to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, source_docs = ask_question(
                    st.session_state.qa_chain,
                    question
                )

            st.write(answer)

            # Deduplicate sources
            seen = set()
            sources = []
            for doc in source_docs:
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "N/A")
                key = f"{source}_page_{page}"
                if key not in seen:
                    sources.append(f"{source} | Page {page}")
                    seen.add(key)

            # Show sources in expander
            with st.expander("📎 Sources"):
                for source in sources:
                    st.write(f"• {source}")

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })