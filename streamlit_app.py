import streamlit as st

from ingestion.load_documents import load_documents
from ingestion.chunking import process_documents
from embeddings.embedding_model import EmbeddingModel
from embeddings.generate_embeddings import embed_texts
from retrieval.similarity import retrieve_top_k
from llm.prompt import build_prompt
from llm.inference import generate_answer
from llm.llm_model import LLM

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="RAG Document QA", layout="wide")
st.title("📚 LLM RAG Document Question Answering")

# ------------------------------
# Load Pipeline (Cached)
# ------------------------------
@st.cache_resource
def load_pipeline():
    docs = load_documents("data/documents")
    chunks = process_documents(docs)

    embedding_model = EmbeddingModel()
    chunk_embeddings = embed_texts(chunks, embedding_model)

    llm = LLM()
    return chunks, chunk_embeddings, embedding_model, llm

chunks, chunk_embeddings, embedding_model, llm = load_pipeline()

# ------------------------------
# User Input
# ------------------------------
question = st.text_input("Ask a question based on the documents:")

if question:
    query_embedding = embedding_model(question)

    retrieved = retrieve_top_k(
        query_embedding,
        chunk_embeddings,
        chunks,
        k=3
    )

    SIMILARITY_THRESHOLD = 0.3

    context_chunks = []
    sources = []

    for score, text, source in retrieved:
        if score >= SIMILARITY_THRESHOLD:
            context_chunks.append(text)
            sources.append(f"{source} (score={score:.2f})")

    if not context_chunks:
        st.warning("I cannot find the answer in the provided documents.")
    else:
        prompt = build_prompt(context_chunks, question)
        answer = generate_answer(prompt, llm)

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📌 Sources")
        for s in sources:
            st.write("-", s)
