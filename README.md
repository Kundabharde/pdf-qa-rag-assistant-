# Chat with PDF (RAG)

A Retrieval-Augmented Generation (RAG) web application that answers questions about an uploaded PDF. Answers are generated only from the document's content and are shown with the source page numbers, which makes them easy to verify.

## Features

- Upload a text-based PDF and ask questions in natural language
- Document chunking with overlap to preserve context across boundaries
- Semantic search over embeddings stored in a Chroma vector store
- Grounded answers: the model is instructed to answer only from the retrieved context
- Source citations with page numbers

## How It Works

1. **Ingestion:** the PDF is read page by page and split into overlapping chunks (1000 characters, 150 overlap).
2. **Indexing:** each chunk is converted into an embedding with Gemini and stored in ChromaDB.
3. **Retrieval:** the question is embedded and the 4 most similar chunks are retrieved.
4. **Generation:** the retrieved chunks and the question are sent to a Gemini model with an instruction to answer only from the provided context.
5. **Citation:** the page numbers of the retrieved chunks are displayed with the answer.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| UI | Streamlit |
| Orchestration | LangChain |
| Vector store | ChromaDB |
| LLM and embeddings | Google Gemini API |
| PDF parsing | pypdf |

## Getting Started

### Prerequisites

- Python 3.12
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com)

### Installation

```bash
git clone https://github.com/Kundabharde/chat-with-pdf-rag.git
cd chat-with-pdf-rag

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a PDF and ask a question.

## Limitations

- Scanned (image-only) PDFs are not supported because no OCR is applied.
- The index is rebuilt for each uploaded document and is not persisted between sessions.
- The free Gemini API tier has a daily request limit.

## Roadmap

- Hybrid search (keyword and semantic) with re-ranking
- A validation step that checks each answer against the retrieved context
- Prompt-injection guardrails and PII masking
- An evaluation script with a labelled question set
