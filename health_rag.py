import os
import uuid
import time
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from google import genai
from llama_index.embeddings.google import GeminiEmbedding
from pinecone import Pinecone, ServerlessSpec
from document_processing import InsuranceDocumentProcessor

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
EMBEDDING_DIM = 768
INDEX_NAME = "health-insurance-rag"
INDEX_HOST = "https://health-insurance-rag-g4jd6mk.svc.aped-4627-b74a.pinecone.io"
TOP_K = 5
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


class HealthInsuranceRAG:
    def __init__(self):
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # existing = self.pc.list_indexes().names()

        # # Create index only if missing
        # if INDEX_NAME not in existing:
        #     self.pc.create_index(
        #         name=INDEX_NAME,
        #         dimension=EMBEDDING_DIM,
        #         metric="cosine",
        #         spec=ServerlessSpec(
        #             cloud="aws",
        #             region=PINECONE_ENV
        #         )
        #     )
        #     time.sleep(2)

        self.index = self.pc.Index(name=INDEX_NAME, host=INDEX_HOST)

        self.processor = InsuranceDocumentProcessor()

        self.embedder = GeminiEmbedding(
            model_name="models/text-embedding-004",
            api_key=GEMINI_API_KEY
        )

    def embed_texts(self, texts: List[str]):
        return self.embedder.get_text_embedding_batch(texts)


    def generate_text(self, prompt: str):
        resp = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"temperature": 0.0}
        )
        return resp.text.strip()

    def process_insurance_documents(self, file_paths: List[str]):


        vectors_to_upsert = []
        total_chunks = 0

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embedder
        )

        for file_path in file_paths:
            file_path = str(file_path)
            file_name = Path(file_path).name
            doc_id = str(uuid.uuid4())

            directory = str(Path(file_path).parent)
            file_docs = SimpleDirectoryReader(
                input_dir=directory,
                required_exts=[".pdf", ".docx", ".txt", ".md"],
                filename_as_id=True
            ).load_data()

            documents = [
                d for d in file_docs
                if d.metadata.get("file_name") == file_name
            ]

            if not documents:
                continue

            li_docs = [
                Document(
                    text=d.text,
                    metadata=d.metadata
                )
                for d in documents
            ]

            nodes = splitter.get_nodes_from_documents(li_docs)
            total_chunks += len(nodes)

            for node in nodes:
                text_chunk = node.get_content()
                embedding = self.embed_texts([text_chunk])[0]

                meta = node.metadata
                page_no = meta.get("page_label", 1)

                vec_id = f"{doc_id}::{page_no}::{node.node_id}"

                metadata = {
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "page_no": page_no,
                    "chunk_index": node.node_id,
                    "text_preview": text_chunk[:300]
                }

                vectors_to_upsert.append({
                    "id": vec_id,
                    "values": embedding,
                    "metadata": metadata
                })

                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []

        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)

        return {
            "status": "ok",
            "message": f"Indexed {len(file_paths)} documents into {total_chunks} semantic chunks"
        }


    def _search(self, query: str):
        query_vec = self.embed_texts([query])[0]

        res = self.index.query(
            vector=query_vec,
            top_k=TOP_K,
            include_metadata=True
        )

        return [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata
            }
            for m in res.matches
        ]
 
    def get_instruction_prompt(self):

        return PromptTemplate(
            """
            You are a professional health insurance assistant.
            Follow these instructions STRICTLY:

            1. Provide accurate and short answers.
            2. Use simple language.
            3. Do NOT hallucinate.
            4. If unsure, say: "I don't have enough information."

            USER QUESTION:
            {question}

            ANSWER:
            """
        )


    def get_context_qa_prompt(self):

        return PromptTemplate(
            """
            You are a health insurance policy expert.

            Use ONLY the context below to answer the user's question.
            If the answer is not present, respond exactly:
            "This information is not covered in the provided policy documents."

            -------------------------
            POLICY CONTEXT:
            {context}
            -------------------------

            USER QUESTION:
            {question}

            IMPORTANT:
            - Keep the answer brief 
            - Mention coverage, limits, exclusions clearly
            - Include ONE citation like:
            "According to {file_name}, Page {page_no}"

            ANSWER:
            """
        )


    def get_refinement_prompt(self):
 
        return PromptTemplate(
            """
            You are refining an insurance answer.

            Original Question:
            {question}

            Existing Answer:
            {existing_answer}

            Additional Context:
            {new_context}

            INSTRUCTIONS:
            - Improve the answer only if new context is useful
            - Do NOT remove correct statements
            - Do NOT hallucinate
            - If the answer cannot be grounded in the document:
                - Respond with a disclaimer
                - Include fallback guidance
                - Example : “The document does not mention this explicitly. Please check the policy’s exclusions section manually.”
            - Maintain clear structure
            - Add citation if needed (Page Number , Section)
            - For broad questions, outputs must follow a structured format:
                - Coverage Details
                   ...
                - Exceptions
                   ...
                - Important Notes
                ...
            - EXAMPLE: "According to Page 26, Section 3.4: “Maternity benefits have a 24-month waiting period.”"

            REFINED ANSWER:
            """
        )
    

    def query_policy(self, question: str):

        matches = self._search(question)
        if not matches:
            return {
                "answer": "No relevant policy information found.",
                "sources": [],
                "confidence": 0.0
            }

        context_text = ""
        citations = []


        for m in matches:
            meta = m["metadata"]

            context_text += f"""
            --- Source: {meta['file_name']} | Page {meta['page_no']} ---
            {meta['text_preview']}
            """

            citations.append({
                "file_name": meta["file_name"],
                "page_no": meta["page_no"],
                "chunk_index": meta["chunk_index"],
                "text_preview": meta["text_preview"],
                "score": m["score"] 
            })


        context_prompt = self.get_context_qa_prompt()
        base_prompt = context_prompt.format(
            context=context_text,
            question=question
        )
        base_answer = self.generate_text(base_prompt)

        refine_prompt = self.get_refinement_prompt()
        refine_final = refine_prompt.format(
            question=question,
            existing_answer=base_answer,
            new_context=context_text
        )
        final_answer = self.generate_text(refine_final)

        confidence = sum([float(m["score"]) for m in matches]) / len(matches)

        return {
            "answer": final_answer,
            "sources": citations,
            "confidence": round(confidence, 3)
        }