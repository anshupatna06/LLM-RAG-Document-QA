import time

from evaluation.retrieval_metrics import recall_at_k
from evaluation.context_coverage import context_coverage
from evaluation.faithfulness import is_faithful
from evaluation.hallucination import grounding_score
from llm.utils import estimate_tokens


def answer_question(
    question,
    top_k,
    threshold,
    pipeline,
    chunk_embeddings,
    chunks,
    embedding_model
):
    start_time = time.time()

    # ------------------------------
    # Query rewriting
    # ------------------------------
    rewritten = pipeline.rewrite_query(question)

    # ------------------------------
    # Retrieval
    # ------------------------------
    t0 = time.time()
    retrieved = pipeline.retrieve(
        rewritten,
        chunk_embeddings,
        chunks,
        top_k
    )
    retrieval_time = time.time() - t0

    used_chunks = []
    retrieval_debug = []

    for idx, (score, text, source) in enumerate(retrieved, start=1):
        used = bool(score >= threshold)

        retrieval_debug.append({
            "rank": int(idx),
            "score": float(score),
            "text": text,
            "source": source,
            "used": used
        })

        if used:
            used_chunks.append(text)

    # ------------------------------
    # If nothing usable retrieved
    # ------------------------------
    if not used_chunks:
        

        return {
            "query": {
                "original": question,
                "rewritten": rewritten
            },
            "answer": "",
        "sources": [],
        "retrieval": {
            "top_k": top_k,
            "threshold": threshold,
            "total_chunks": len(chunks),
            "retrieved_chunks": len(retrieved),
            "used_chunks": 0,
            "chunks": retrieval_debug
        },
        "failure": {
            "type": "BELOW_THRESHOLD",
            "reason": "No retrieved chunks passed the similarity threshold",
            "threshold": threshold,
            "max_score": max([c["score"] for c in retrieval_debug], default=0.0)
        },
        "metrics": {
            "recall_at_k": 0.0,
            "context_coverage": 0.0,
            "faithful": False,
            "grounding_score": 0.0
        },
        "performance": {
            "latency": {
                "total_sec": round(time.time() - start_time, 3)
            },
            "cost": {
                "total_tokens": 0,
                "estimated_cost_usd": 0.0
            }
        }
    }

    # ------------------------------
    # LLM Answering
    # ------------------------------
    t1 = time.time()
    answer = pipeline.answer(question, retrieved)
    llm_time = time.time() - t1

    total_latency = time.time() - start_time

    # ------------------------------
    # Metrics (FORCE Python types)
    # ------------------------------
    recall = float(recall_at_k(retrieved))
    coverage = float(context_coverage(answer, used_chunks))
    faithful = bool(is_faithful(coverage))
    grounding = float(grounding_score(answer, used_chunks, embedding_model))

    # ------------------------------
    # Cost estimation
    # ------------------------------
    prompt_tokens = int(estimate_tokens(" ".join(used_chunks)))
    completion_tokens = int(estimate_tokens(answer))
    total_tokens = int(prompt_tokens + completion_tokens)
    estimated_cost = float(total_tokens * 0.000002)

    # ------------------------------
    # Final response (JSON SAFE)
    # ------------------------------
    return {
        "query": {
            "original": question,
            "rewritten": rewritten
        },
        "answer": answer,
        "sources": list(
            set(c["source"] for c in retrieval_debug if c["used"])
        ),
        "retrieval": {
            "top_k": int(top_k),
            "threshold": float(threshold),
            "total_chunks": int(len(chunks)),
            "retrieved_chunks": int(len(retrieved)),
            "used_chunks": int(len(used_chunks)),
            "chunks": retrieval_debug
        },
        "metrics": {
            "recall_at_k": recall,
            "context_coverage": coverage,
            "faithful": faithful,
            "grounding_score": grounding
        },
        "performance": {
            "latency": {
                "retrieval_sec": float(round(retrieval_time, 3)),
                "llm_sec": float(round(llm_time, 3)),
                "total_sec": float(round(total_latency, 3))
            },
            "cost": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": float(round(estimated_cost, 6))
            }
        }
    }
