import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from health_rag import HealthInsuranceRAG
from document_processing import InsuranceDocumentProcessor

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

USER_UPLOAD = os.getenv("USER_UPLOAD", "user_upload")
os.makedirs(USER_UPLOAD, exist_ok=True)

app = FastAPI(title="Health Insurance RAG (Gemini + Pinecone)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = InsuranceDocumentProcessor()

try:
    rag = HealthInsuranceRAG()
except Exception as e:
    rag = None
    init_error = str(e)
else:
    init_error = None


class QuestionReq(BaseModel):
    question: str


@app.post("/system-upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):

    if rag is None:
        raise HTTPException(status_code=500, detail=f"RAG system not initialized: {init_error}")

    saved_paths = []

    for file in files:
        filename = file.filename
        if not filename or not filename.lower().endswith((".pdf", ".docx", ".txt", ".md")):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

        unique_name = f"{uuid.uuid4().hex}_{filename}"
        path = os.path.join(UPLOAD_DIR, unique_name)

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        valid, msg = processor.validate_insurance_document(path)
        if not valid:
            try:
                os.remove(path)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=f"Invalid insurance document: {msg}")

        saved_paths.append(path)

    result = rag.process_insurance_documents(saved_paths)
    return {"status": "success", "detail": result}

@app.post("/ask-question")
async def ask_question(req: QuestionReq):
    if rag is None:
        raise HTTPException(status_code=500, detail=f"RAG system not initialized: {init_error}")

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    resp = rag.query_policy(q)
    return resp


@app.get("/")
def root():
    return {"message": "Health Insurance RAG (Gemini + Pinecone) is running."}
