import argparse

from ingestion.load_documents import load_documents
from ingestion.chunking import process_documents
from embeddings.embedding_model import EmbeddingModel
from embeddings.generate_embeddings import embed_texts
from retrieval.similarity import retrieve_top_k
from llm.llm_model import LLM
from rag_core.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG CLI Interface")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval")

    args = parser.parse_args()

    # Load pipeline
    docs = load_documents("data/documents")
    chunks = process_documents(docs)

    embedding_model = EmbeddingModel()
    chunk_embeddings = embed_texts(chunks, embedding_model)

    llm = LLM()
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        llm=llm,
        retriever=retrieve_top_k
    )

    retrieved = pipeline.retrieve(
        args.query,
        chunk_embeddings,
        chunks,
        args.k
    )

    answer = pipeline.answer(args.query, retrieved)

    print("\nAnswer:\n", answer)
    print("\nSources:")
    for _, _, source in retrieved:
        print("-", source)


if __name__ == "__main__":
    main()
