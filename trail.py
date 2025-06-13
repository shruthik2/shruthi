import os
import tempfile
import streamlit as st

# LangChain & Gemini
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ─────────────────────────────────────────────────────
# Config: Gemini API
# ─────────────────────────────────────────────────────
GOOGLE_API_KEY = "AIzaSyDrA2-nxoGG5VoupuYhpXcOrQiE0w2tqUM"  # Replace with your actual key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

LLM_MODEL_NAME = "gemini-2.0-flash"
EMBED_MODEL = "models/embedding-001"

llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

# ─────────────────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use only the provided CONTEXT to answer the QUESTION. "
               "If the answer isn't in the context, say 'I don’t know.'"),
    ("user", "CONTEXT:\n{context}\n\nQUESTION: {question}")
])

# ─────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Chat with Your Document — Gemini RAG Demo")

with st.sidebar:
    st.header("📄 Upload your document")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

# ─────────────────────────────────────────────────────
# File Processing: Load → Split → Embed → FAISS
# ─────────────────────────────────────────────────────
def process_file(file_) -> FAISS:
    suffix = ".pdf" if file_.type == "application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_.read())
        path = tmp.name

    docs = PyPDFLoader(path).load() if suffix == ".pdf" else TextLoader(path, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    return FAISS.from_documents(splits, embeddings)

if uploaded_file and "vectorstore" not in st.session_state:
    with st.spinner("📚 Processing document..."):
        st.session_state.vectorstore = process_file(uploaded_file)
    st.success("✅ Document indexed. Ask your question!")

# ─────────────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────────────
if "vectorstore" in st.session_state:
    question = st.text_input("💬 Ask a question based on your document:")

    if st.button("🔍 Get Answer") and question.strip():
        try:
            retriever = st.session_state.vectorstore.as_retriever(k=4)
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)

            chain = RAG_PROMPT | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})

            st.success("✅ Answer:")
            st.write(answer)

            with st.expander("📄 Retrieved Document Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}:**\n{doc.page_content}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("📤 Please upload a document to begin.")
