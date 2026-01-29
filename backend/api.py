from fastapi import FastAPI, UploadFile, File
import os
from pydantic import BaseModel

from backend.schemas import QueryRequest, QueryResponse
from backend.rag_service import answer_question
from backend.state import state, DATA_DIR

from retrieval.similarity import retrieve_top_k
from llm.llm_model import LLM
from rag_core.pipeline import RAGPipeline

# ------------------------------
# App
# ------------------------------
app = FastAPI(title="RAG API")

# ------------------------------
# Pipeline (no document loading here)
# ------------------------------
llm = LLM()
pipeline = RAGPipeline(
    embedding_model=state.embedding_model,
    llm=llm,
    retriever=retrieve_top_k
)

# ------------------------------
# Root
# ------------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running", "docs": "/docs"}

# ------------------------------
# Upload document
# ------------------------------
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    state.reload()  # rebuild chunks + embeddings
    return {"message": f"{file.filename} uploaded and indexed successfully"}

# ------------------------------
# List documents
# ------------------------------
@app.get("/documents")
def list_documents():
    return {"documents": os.listdir(DATA_DIR)}

# ------------------------------
# Delete document
# ------------------------------
class DeleteRequest(BaseModel):
    filename: str

@app.delete("/documents")
def delete_document(req: DeleteRequest):
    file_path = os.path.join(DATA_DIR, req.filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    os.remove(file_path)
    state.reload()
    return {"message": f"{req.filename} deleted successfully"}

# ------------------------------
# Query endpoint (FINAL)
# ------------------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Thin API layer.
    All logic lives inside rag_service.answer_question
    """

    result = answer_question(
        request.question,
        request.top_k,
        request.threshold,
        pipeline,
        state.chunk_embeddings,
        state.chunks,
        state.embedding_model
    )

    return result
