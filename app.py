# app.py
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from agent import executor
from backend.state import state, DATA_DIR
from backend.schemas import QueryRequest
from rag_core.pipeline import RAGPipeline
from retrieval.similarity import retrieve_top_k
from llm.llm_model import LLM
from agent.agent_controller import RAGAgent
from agent.executor import AgentExecutor
from business.welcome_message import generate_welcome_message
from business.business_config import get_business_config
from fastapi.middleware.cors import CORSMiddleware
import shutil
from business.router import detect_business

import math
def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj



app = FastAPI(title="RAG + Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/{business_id}/welcome")
def get_welcome(business_id: str):

    config = get_business_config(business_id)
    welcome_text = generate_welcome_message(config)

    return {
        "message": welcome_text,
        "suggestions": [
            "Is breakfast included?",
            "What is check-in time?",
            "What facilities are available?"
        ]
    }


# ------------------------------
# Upload document(chnage it to include business-id for multi tenant architecture)
# ------------------------------
DATA_DIR = "data/documents"

@app.post("/{business_id}/{client_id}/upload")
def upload_document(
    business_id: str,
    client_id: str,
    file: UploadFile = File(...)
):

    folder = os.path.join(DATA_DIR, business_id, client_id)

    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, file.filename)
    print("Saving file to:", file_path)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #state.reload()

    return {
        "message": f"{file.filename} uploaded for {business_id}/{client_id}"
    }


@app.post("/reindex")
def reindex():
    state.reload()
    return {"status": "embeddings rebuilt"}

# ------------------------------
# List documents
# ------------------------------
# @app.get("/documents")
# def list_documents():

#     result = {}

#     for business in os.listdir(DATA_DIR):

#         path = os.path.join(DATA_DIR, business)

#         if os.path.isdir(path):
#             result[business] = os.listdir(path)

#     return result

@app.get("/documents")
def list_documents():

    result = {}

    for business in os.listdir(DATA_DIR):

        business_path = os.path.join(DATA_DIR, business)

        if os.path.isdir(business_path):

            result[business] = []

            for client in os.listdir(business_path):

                client_path = os.path.join(business_path, client)

                if os.path.isdir(client_path):
                    result[business].append(client)

    return result

# ------------------------------
# Delete document
# ------------------------------
class DeleteRequest(BaseModel):
    filename: str
    business: str


@app.delete("/documents")
def delete_document(req: DeleteRequest):

    file_path = os.path.join(DATA_DIR, req.business, req.filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    os.remove(file_path)

    state.reload()

    return {"message": f"{req.filename} deleted from {req.business}"}



@app.post("/{business_id}/{client_id}/query")
def query_with_agent(
    business_id: str,
    client_id: str,
    req: QueryRequest
):

     # auto detect if question belongs to another domain
    detected = detect_business(req.question)

    # if detected in state.indices:
    #     business_id = detected

    agent = RAGAgent(pipeline)

    # if business_id not in state.indices:
    #     return {
    #         "answer": "No documents uploaded for this assistant yet.",
    #         "suggestions": []
    #     }

    # if client_id not in state.indices.get(business_id, {}):
    #         return {"answer": "No documents found for this client."}

    index = state.get_index(business_id, client_id)

    if not index:
        return {"answer": "No documents found for this client."}


    # executor = AgentExecutor(
    #     agent,
    #     index["fine_chunks"],
    #     index["coarse_chunks"],
    #     index["fine_embeddings"],
    #     index["coarse_embeddings"],
    #     index["fine_bm25"],
    #     index["coarse_bm25"]
    # )
    executor = AgentExecutor(agent, index)

    result = executor.run(req.question, business_id)

    if result is None:
        print("❌ ERROR: executor returned None")
        return {
            "answer": "Something went wrong. Please try again.",
            "sources": [],
            "retrieval": {},
        }

    # 🔥 FIX: ensure answer is string
    if isinstance(result.get("answer"), tuple):
        result["answer"] = result["answer"][0]

    if not result.get("answer"):
        result["answer"] = "I cannot find this information in the provided documents."

    safe_result = sanitize_for_json(result)

    return safe_result

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000)