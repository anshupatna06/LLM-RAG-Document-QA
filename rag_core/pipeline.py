# rag_core/pipeline.py

from llm.prompt import build_prompt
from llm.inference import generate_answer

class RAGPipeline:
    def __init__(self, embedding_model, llm, retriever):
        self.embedding_model = embedding_model
        self.llm = llm
        self.retriever = retriever
        
    def rewrite_query(self, question: str) -> str:
        q = question.lower().strip()

        if q.startswith("do you know"):
            q = q.replace("do you know", "explain")
        if q.startswith("what do you mean by"):
            q = q.replace("what do you mean by", "explain")
        if q.startswith("tell me about"):
            q = q.replace("tell me about", "explain")

        return q
    
    def retrieve(self, query, chunk_embeddings, chunks, k):
        query_emb = self.embedding_model(query)
        return self.retriever(query_emb, chunk_embeddings, chunks, k)

    def answer(self, question, retrieved_chunks):
        context = [text for _, text, _ in retrieved_chunks]
        prompt = build_prompt(context, question)
        return generate_answer(prompt, self.llm)
