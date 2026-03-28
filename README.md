📄 DocMind — Your AI Document Assistant

DocMind is an AI-powered document Q&A assistant that lets you chat with your PDFs and files. Upload research papers, textbooks, or any document and ask questions in natural language to get instant, context-aware answers.

🚀 Features
- 📂 Upload one or multiple documents
- 💬 Ask questions in plain English
- 🧠 Context-aware answers using LLMs
- 🔍 Semantic search with vector embeddings
- ⚡ Fast and interactive UI with Streamlit

🛠️ Tech Stack
- LLM Framework: LangChain
- Model Provider: Groq (LLaMA 3.3 70B)
- Embeddings: HuggingFace
- Vector Database: FAISS
- Frontend/UI: Streamlit

📦 Installation & Setup

Follow these steps to run the project locally:

1. Clone the repository
git clone https://github.com/Palakjn0965/DocMind---AI-Document-Assistant
-cd docmind

2. Create a virtual environment
- python -m venv .venv
- source .venv/bin/activate     # On macOS/Linux
- .venv\Scripts\activate        # On Windows

4. Install dependencies
- uv sync

5. Configure environment variables

Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY=your_key_here

6. Run the app
- streamlit run app.py
- 
🧪 How It Works
- Upload your document(s)
- Text is split into chunks
- Embeddings are generated using HuggingFace
- Stored in a FAISS vector database
- User queries are matched with relevant chunks
- Groq LLaMA model generates accurate answers

⚠️ Limitations
- Performance depends on document size and chunking
- Requires internet for API calls
- Accuracy depends on embedding quality and model responses
