import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import tempfile
import os
import pickle


# ------------------------------- Streamlit UI -------------------------------
st.set_page_config(page_title="GenAI Doc Assistant", layout="wide")
st.title("📄 GenAI Document Assistant 🌍")
st.markdown("Upload **PDF**, **TXT**, or **DOCX** and ask questions!")


# -------------------------- Utility Functions --------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text


def extract_docx_text(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


# ------------------------------- Main App -------------------------------
def main():

    uploaded_file = st.file_uploader("📁 Upload document", type=["pdf", "txt", "docx"])

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()

        # Read File
        if ext == "pdf":
            text = extract_text_from_pdf(uploaded_file)

        elif ext == "txt":
            text = uploaded_file.read().decode("utf-8")

        elif ext == "docx":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            text = extract_docx_text(tmp_path)

        else:
            st.error("Unsupported file format.")
            return

        # ------------------- Text Splitting -------------------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_text(text)

        store_name = uploaded_file.name.replace(".", "_")

        # ------------------- Vector Store Load or Create -------------------
        if os.path.exists(f"{store_name}_chunks.pkl"):
            with open(f"{store_name}_chunks.pkl", "rb") as f:
                chunks = pickle.load(f)

            embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        else:
            embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}_chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # ------------------- LCEL RAG Pipeline -------------------
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You answer questions ONLY using the document."),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        llm = ChatGroq(model="llama-3.1-8b-instant")

        rag_chain = (
            RunnableParallel(
                context=retriever,
                question=RunnablePassthrough()
            )
            | prompt
            | llm
        )

        # ------------------- Ask Question -------------------
        query = st.text_input("🗣️ Ask anything:")

        if query:
            with st.spinner("🔍 Searching..."):
                response = rag_chain.invoke(query)

            st.write("🤖 **Response:**")
            st.success(response.content)


if __name__ == "__main__":
    main()


# ------------------------------- Sidebar Footer -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💻 **Built by Ankit Shah**")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ank-it-shah/) | [GitHub](https://github.com/ankitshah074)")
