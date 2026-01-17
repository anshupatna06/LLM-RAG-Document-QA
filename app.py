from embeddings.generate_embeddings import embed_texts
from embeddings.embedding_model import EmbeddingModel

from ingestion.load_documents import load_documents
from ingestion.chunking import process_documents

from retrieval.similarity import retrieve_top_k
from llm.prompt import build_prompt
from llm.inference import generate_answer
from llm.llm_model import LLM

from evaluation.retrieval_metrics import recall_at_k
from evaluation.context_coverage import context_coverage
from evaluation.faithfulness import is_faithful


# Initialize models ONCE
embedding_model = EmbeddingModel()
llm = LLM()


# Load and process documents
docs = load_documents("data/documents")
chunks = process_documents(docs)

# Generate embeddings for document chunks
chunk_embeddings = embed_texts(chunks, embedding_model)

# User query
question = input("Ask a question: ")

# Embed the query
embedding_model = EmbeddingModel()
chunk_embeddings = embed_texts(chunks, embedding_model)
query_embedding = embedding_model(question)


# Retrieve relevant chunks
SIMILARITY_THRESHOLD = 0.3

retrieved = retrieve_top_k(
    query_embedding,
    chunk_embeddings,
    chunks,
    k=3
)

context_chunks = []
sources = []

for score, text, source in retrieved:
    if score >= SIMILARITY_THRESHOLD:
        context_chunks.append(text)      # ✅ ONLY text
        sources.append(f"{source} (score={score:.2f})")

if not context_chunks:
    print("I cannot find the answer in the provided documents.")
    exit()




# Build prompt and generate answer
prompt = build_prompt(context_chunks, question)
answer = generate_answer(prompt, llm)

print("\nAnswer:\n")
print(answer)

print("\nSources:")
for s in sources:
    print("-", s)

# Evaluation
recall = recall_at_k(retrieved)
coverage = context_coverage(answer, context_chunks)
faithful = is_faithful(coverage)

print("\nEvaluation Metrics:")
print(f"Recall@K: {recall}")
print(f"Context Coverage: {coverage:.2f}")
print(f"Faithful Answer: {faithful}")
