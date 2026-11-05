import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄")
st.title("📄 PDF Q&A Assistant")
st.caption("Ask questions about a PDF. Answers are grounded in the document and cite their source pages.")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Add it to a .env file and restart the app.")
    st.stop()


def build_vectorstore(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    if not chunks:
        return None, 0

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(chunks, embeddings), len(chunks)


def answer_question(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = "Answer the question using only the context below. "
    prompt += "If the answer is not in the context, reply exactly: This information is not available in the document.\n\n"
    prompt += f"Context:\n{context}\n\n"
    prompt += f"Question: {question}\n"
    prompt += "Answer:"

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    return llm.invoke(prompt).content, docs


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Processing the document..."):
            try:
                vectorstore, n_chunks = build_vectorstore(uploaded_file)
            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()
        if vectorstore is None:
            st.error("No readable text was found. Scanned (image-only) PDFs are not supported.")
            st.stop()
        st.session_state.vectorstore = vectorstore
        st.session_state.file_name = uploaded_file.name
        st.session_state.n_chunks = n_chunks

    st.success(f"Document indexed ({st.session_state.n_chunks} chunks). You can now ask questions.")

    question = st.text_input("Your question")

    if question:
        with st.spinner("Generating answer..."):
            try:
                answer, docs = answer_question(st.session_state.vectorstore, question)
            except Exception as exc:
                message = str(exc)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    st.error("The Gemini API quota has been reached. Please try again later.")
                else:
                    st.error(f"Could not generate an answer: {message}")
                st.stop()

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
        st.write("Pages: " + ", ".join(str(p) for p in pages))
        with st.expander("View retrieved passages"):
            for d in docs:
                st.markdown(f"**Page {d.metadata.get('page', 0) + 1}**")
                st.write(d.page_content)