# Health Insurance Policy Enquiry System (RAG)

## Overview

This repository implements a Retrieval-Augmented Generation (RAG) system that answers user questions about health insurance policies using semantic search combined with LLM reasoning.

The system leverages the following technologies:

- **Google Gemini** (via `google-genai`) for text embeddings and LLM responses
- **LlamaIndex** for intelligent document parsing, semantic splitting, and embedding utilities
- **Pinecone Vector Database** for persistent vector storage and fast similarity search
- **FastAPI** for backend services (document upload and query endpoints)
- **Streamlit** for lightweight user interface

### System Workflow

1. Upload a policy document (PDF/DOCX/TXT/MD)
2. System extracts text, creates semantic chunks, and generates embeddings
3. Embeddings stored in Pinecone with metadata (file, page, chunk)
4. User asks a question which is embedded and used to query Pinecone for relevant chunks
5. Retrieved chunks are passed into a prompt template and Gemini generates a concise answer with citation

---

## Project Structure

```
.
├── document_processing.py    # Extracts text, validates, preprocesses
├── health_rag.py             # RAG pipeline: embeddings, Pinecone, queries, prompts
├── main.py                   # FastAPI backend (upload + ask-question)
├── streamlit_app.py          # Frontend UI
├── requirements.txt
├── .env.example
└── README.md

```

---

## Quick Start (Local)

### 1. Clone Repository

```bash
git clone <repo-url>
cd <repo-dir>

```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

```

### 3. Install Requirements

```bash
pip install -r requirements.txt

```

### 4. Create .env File

```
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-east-1
PINECONE_INDEX_NAME=health-insurance-rag
EMBEDDING_DIM=1536
UPLOAD_DIR=uploaded_docs
USER_UPLOAD=user_upload
INDEX_HOST=<optional>

```

### 5. Start FastAPI Backend

```bash
uvicorn main:app --reload

```

### 6. Start Streamlit Frontend

```bash
streamlit run streamlit_app.py

```

### 7. Access Application

- Frontend: http://localhost:8501
- API documentation: http://127.0.0.1:8000/docs

---

## Environment Variables

| **Key** | **Description** |
| --- | --- |
| GEMINI_API_KEY | Required — embedding and LLM |
| PINECONE_API_KEY | Required — vector DB |
| PINECONE_ENV | Pinecone region |
| PINECONE_INDEX_NAME | Index name |
| EMBEDDING_DIM | Should match embedding model |
| UPLOAD_DIR | Location for system uploads |
| USER_UPLOAD | User directory |
| INDEX_HOST | Custom Pinecone host |

---

## How the application works

### 1. Upload Documents

Supported formats: .pdf, .docx, .txt, .md

### 2. System Processing

- Validates document
- Extracts text
- Splits into semantic nodes
- Generates embeddings
- Inserts vectors into Pinecone

### 3. Ask Questions

The system returns:

- Short answer
- Citation (file + page + section (if available))
- Confidence score

---

## Approach

This project uses a RAG (Retrieval-Augmented Generation) workflow:

1. **Retrieve** relevant information from the stored vector database
2. **Augment** the user question with retrieved context
3. **Generate** a concise LLM answer strictly from the context

This prevents hallucination and ensures the answer is grounded in the uploaded policy documents.

---

## Architecture (High-Level)

```
USER → Streamlit UI
       ↓
FastAPI Backend → Validates & saves documents
       ↓
Document Processing → Extract text
       ↓
Semantic Splitter → Create smart chunks
       ↓
Embeddings (Gemini)
       ↓
Pinecone Vector DB
       ↓
Question Asked
       ↓
Embed Question → Query Pinecone
       ↓
Retrieve Top-K Chunks
       ↓
Prompts built using templates
       ↓
LLM (Gemini) generates grounded answer
       ↓
Return answer + citations + confidence

```

---

## Evaluations

### 1. Embedding Quality

We validated that:

- SemanticSplitter produced smaller, meaningful nodes
- Gemini-004 embeddings gave better relevance scores
- Pinecone returned correct matches using cosine similarity

### 2. Prompt Performance

We compared:

- Simple prompts resulted in long, verbose answers
- Structured prompts produced concise and factual answers
- Refinement prompt improved factual precision

### 3. Chunking Evaluation

Approaches tested:

- Semantic splitting (95th percentile breakpoints)

Semantic splitter performed better and was used in final design, is used because instead of chunking based on defined number of charaters Semantic splitting captures topic/semantic meaning of every chunk. It chunks with respect to the semantic meaning when the next line is more than 5% irrelevent it cuts the chunk from that line.

---

## Prompt Engineering

We used three types of prompts:

### 1. Instruction Prompt

Short, strict rules for clear output.

### 2. Context QA Prompt

Injects retrieved chunks.

### 3. Refinement Prompt

Improves initial answer while staying truthful.

All three are implemented in health_rag.py.

---
LINK FOR NOTION DOCUMENTATION :

https://alkaline-sundae-d57.notion.site/Health-Insurance-System-RAG-2ad8140798f280eba607c854ed98a76f
