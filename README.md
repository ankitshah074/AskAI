# 🧠 AskAI Document Assistant

**GenAI Document Assistant** is a AI-powered application that allows users to upload documents (PDF, DOCX) and ask questions in any major language. The assistant intelligently understands the document, retrieves relevant context, and answers questions using advanced language models and vector similarity search.

## 📸 Screenshot

![App Screenshot](https://github.com/ankitshah074/AskAI/blob/main/AskAI.png)

check it- https://askai-doc.streamlit.app/

---

## 🌟 Features

📁 Multi-file support — PDF, DOCX, TXT

🔍 Semantic search using MiniLM embeddings

🧩 Chunking using LangChain’s RecursiveCharacterTextSplitter

📦 Vector storage via FAISS or Chroma

⚡ Fast LLM inference using Groq’s LLaMA-3.1 models

❓ Ask questions directly from document content

💾 Local chunk caching for faster repeated uploads

🌐 Streamlit-based UI for easy interaction 

---

## 🛠️ Tech Stack

Languages: Python

Frameworks: LangChain, Streamlit

AI Models: LLaMA 3.1 (Groq API), MiniLM Embeddings

Vector DB: FAISS, Chroma

Libraries: PyPDF2, python-docx, SentenceTransformers 

---
## 📂 Project Architecture
Upload File → Extract Text → Chunk Text → Create Embeddings
          → Store in Vector DB → Retrieve Relevant Chunks
          → Build Prompt → LLM (Groq) → Final Answer


## 🧠 How It Works

1️⃣ Upload Document

User uploads a PDF, TXT, or DOCX file.

2️⃣ Text Extraction

PyPDF2 → for PDFs

python-docx → for DOCX

decode() → for text files

3️⃣ Chunking the Document

Document is split into overlapping chunks (1000 tokens, 200 overlap) to preserve context.

4️⃣ Embedding Generation

SentenceTransformer (MiniLM-L6-v2) creates semantic embeddings for each chunk.

5️⃣ Vector Store Creation

Vectors are stored in FAISS or Chroma DB.
Local caching improves repeated performance.

6️⃣ Retriever Logic

Top 3 relevant chunks are retrieved using vector similarity search.

7️⃣ LLM Response (RAG)

Chunks + question are passed to Groq's LLaMA model to produce accurate, grounded answers.

8️⃣ Streamlit UI

Interactive interface for uploading files and asking queries.

## Example Use Cases
📚 Students: Ask questions about lecture notes or study material

🧑‍💼 Professionals: Extract summaries from business reports or whitepapers

👨‍⚖️ Legal: Query long contracts or case files

📊 Research: Analyze papers or data documentation


---

### 📦 Local Deployment

```bash
git clone https://github.com/ankitshah074/AskAI.git
cd AskAI
pip install -r requirements.txt
GROQ_API_KEY=your_key_here //file .env
streamlit run app.py
