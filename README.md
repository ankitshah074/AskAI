# 🧠 AskAI Document Assistant

**GenAI Document Assistant** is a AI-powered application that allows users to upload documents (PDF, DOCX) and ask questions in any major language. The assistant intelligently understands the document, retrieves relevant context, and answers questions using advanced language models and vector similarity search.

## 📸 Screenshot

![App Screenshot](https://github.com/ankitshah074/AskAI/blob/main/AskAI.png)

check it- https://askai-doc.streamlit.app/

---

## 🌟 Features

✅ Supports **PDF**, **Word (.docx)** files  
✅ **Q&A** — ask any questions related to your document.  
✅ Uses **sentence embeddings** + **vector search (FAISS)** for intelligent context retrieval  
✅ Powered by **Hugging Face Transformers** and **LangChain**  
✅ Built with **Streamlit** for fast, clean web app deployment  

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` from `sentence-transformers`  
- **QA**: `Groq LLM (Llama3)` Groq LLM (Llama3)	Runs the actual Large Language Model for answering questions  
- **Vector Search**: FAISS (Facebook AI Similarity Search)  
- **Text Processing**: LangChain’s text splitter for chunking long documents  

---

## 🧠 How It Works

1. **Upload a Document**  
   Upload `.pdf`, `.txt`, or `.docx` format. Text is extracted and chunked into small segments.

2. **Embedding & Indexing**  
   Each chunk is converted into a dense vector using a multilingual sentence transformer and stored in a FAISS index.

3. **Ask a Question**  
   The user submits a query in any language. The query is embedded, and the assistant finds the most relevant document chunks.

4. **Answer Generation**  
   LLM with **Groq LLaMA**  model analyzes the context and returns the most accurate answer to the user.

## Example Use Cases
📚 Students: Ask questions about lecture notes or study material
🧑‍💼 Professionals: Extract summaries from business reports or whitepapers
👨‍⚖️ Legal: Query long contracts or case files
📊 Research: Analyze papers or data documentation


---

## 🚀 Deployment

### 📦 Local Deployment

```bash
git clone https://github.com/ankitshah074/AskAI.git
cd AskAI
pip install -r requirements.txt
streamlit run app.py
