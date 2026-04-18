# rag_core/pipeline.py

from llm.prompt import build_prompt
from llm.inference import generate_answer
import re


class RAGPipeline:
    def __init__(self, embedding_model, llm, retriever, bm25=None):
        self.embedding_model = None
        self.llm = llm
        self.retriever = retriever
        self.bm25 = bm25

    def rewrite_query(self, question: str, memory=None) -> str:

        q = question.lower().strip()

        # remove trailing filler words
        q = re.sub(r"\b(in|at|the|a|an)$", "", q)

        # conversational grounding
        if memory and memory.conversation.get("last_topic"):
            if len(q.split()) <= 3:
                q = memory.conversation["last_topic"] + " " + q

        if q.startswith("do you know"):
            q = q.replace("do you know", "explain")

        if q.startswith("what do you mean by"):
            q = q.replace("what do you mean by", "explain")

        if q.startswith("tell me about"):
            q = q.replace("tell me about", "explain")

        return q.strip()
    

    # def normalize_sub_query(p):

    #     p = p.strip()

    #     if any(p.startswith(w) for w in ["what","is","are","do","does","can"]):
    #         return p

    #     # 🔥 SINGLE WORD → convert
    #     if len(p.split()) == 1:
    #         return f"do you offer {p}"

    #     return p
    

    # def split_multi_query(self, question):

    #     q = question.lower()

    #     connectors = [" and ", " also ", " & ", ","]

    #     for c in connectors:

    #         if c in q:

    #             parts = [p.strip() for p in q.split(c)]

    #             # keep only meaningful parts
    #             valid = []

    #             for p in parts:

    #                 # ignore short fragments
    #                 if len(p.split()) < 2:
    #                     continue

    #                 # 🔥 accept noun phrases also
    #                 if len(p.split()) >= 1:
    #                     valid.append(p)

    #                 # must contain a query indicator
    #                 # if any(w in p for w in [
    #                 #     "what","which","is","are","do","does","where","when","how"
    #                 # ]):
    #                 #     valid.append(p)

    #             if len(valid) >= 2:
    #                 return valid
                
    #             def normalize_sub_query(p):

    #                 p = p.strip()

    #                 # 🔥 already a question → keep
    #                 if any(p.startswith(w) for w in ["what","is","are","do","does","can"]):
    #                     return p

    #                 # 🔥 otherwise convert to binary query
    #                     return f"do you offer {p}"
                
    #             valid = [normalize_sub_query(p) for p in valid]

    #     return [q]

    def split_multi_query(self, question):

        q = question.lower()

        connectors = [" and ", " also ", " & ", ","]

        for c in connectors:

            if c in q:

                parts = [p.strip() for p in q.split(c)]

                # 🔥 DO NOT REMOVE SINGLE WORDS
                parts = [p for p in parts if len(p) > 1]

                if len(parts) >= 2:
                    return parts

        return [q]

    def retrieve(self, query, chunk_embeddings, chunks, k):

        query_emb = self.embedding_model(query)

        return self.retriever(
            query_emb,
            chunk_embeddings,
            chunks,
            k
        )

    def answer(self, question, retrieved_chunks, system_prompt=None):

        # reduce token overload
        MAX_CONTEXT = 350

        context = []

        for _, text, _ in retrieved_chunks[:3]:
            context.append(text[:MAX_CONTEXT])

        # safety fallback
        if not context and retrieved_chunks:
            context.append(retrieved_chunks[0][1][:MAX_CONTEXT])
            
        prompt = build_prompt(
            context_chunks=context,
            question=question,
            system_prompt=system_prompt
        )
        print("Prompt length:", len(prompt)) # DEBUG TOOL

        return generate_answer(context, question)