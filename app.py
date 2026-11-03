import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
	ChatGoogleGenerativeAI,
	GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
	if (
		"vectorstore" not in st.session_state
		or st.session_state.get("file_name") != uploaded_file.name
	):
		tmp_path = None
		with st.spinner("Processing your PDF..."):
			try:
				with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
					tmp.write(uploaded_file.getvalue())
					tmp_path = tmp.name

				docs = PyPDFLoader(tmp_path).load()
				splitter = RecursiveCharacterTextSplitter(
					chunk_size=1000, chunk_overlap=150
				)
				chunks = splitter.split_documents(docs)

				embeddings = GoogleGenerativeAIEmbeddings(
					model="models/gemini-embedding-001"
				)
				st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
				st.session_state.file_name = uploaded_file.name
			finally:
				if tmp_path and os.path.exists(tmp_path):
					os.remove(tmp_path)

		st.success(f"Created {len(chunks)} chunks. Ask your question below!")

	question = st.text_input("Ask a question about your PDF:")

	if question:
		with st.spinner("Thinking..."):
			retriever = st.session_state.vectorstore.as_retriever(
				search_kwargs={"k": 4}
			)
			relevant_docs = retriever.invoke(question)
			context = "\n\n".join(doc.page_content for doc in relevant_docs)

			llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
			prompt = (
				"Answer the question only using the context below. "
				"If the answer is not in the context, say: This information is not in the PDF.\n\n"
				f"Context:\n{context}\n\n"
				f"Question: {question}\n"
				"Answer:"
			)
			answer = llm.invoke(prompt).content

		st.subheader("Answer")
		st.write(answer)

		st.subheader("Sources")
		for doc in relevant_docs:
			st.caption(f"Page {doc.metadata.get('page', 0) + 1}")
