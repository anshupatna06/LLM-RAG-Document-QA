# agent/executor.py
import time

from agent.memory import AgentMemory
from agent.planner import plan
from agent.tools.retrieve import retrieve_chunks, normalize_query
from agent.tools.summarize import summarize_chunks
from agent.tools.validate import validate_answer
from agent.tools.refuse import refuse
from agent.tools.clarify import clarify
from agent.tools.suggest import generate_suggestions

from business.business_config import get_business_config
from agent.utils.intent import detect_intent
from agent.utils.query_classifier import is_time_query, is_list_query, is_binary_question, is_feature_query
# from rag_core import normalize_sub_query

DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.30


# -----------------------------
# 🔥 SIMPLE IN-MEMORY CACHE
# -----------------------------
# CACHE = {}

# def make_cache_key(query, business_id):
#     return f"{business_id}:{query.lower().strip()}"


# def get_cached_response(query, business_id):
#     key = make_cache_key(query, business_id)
#     return CACHE.get(key)


# def set_cached_response(query, business_id, result):
#     key = make_cache_key(query, business_id)
#     CACHE[key] = result


import time
import re
from collections import OrderedDict

CACHE = OrderedDict()
CACHE_TTL = 300
MAX_CACHE_SIZE = 100


def make_cache_key(query, business_id):
    q = query.lower()
    q = re.sub(r"\s+", " ", q).strip()
    return f"{business_id}:{q}"


def get_cached_response(query, business_id):
    key = make_cache_key(query, business_id)

    data = CACHE.get(key)

    if not data:
        return None

    # TTL check
    if time.time() - data["timestamp"] > CACHE_TTL:
        del CACHE[key]
        return None

    # mark as recently used
    CACHE.move_to_end(key)

    return data["result"]


def set_cached_response(query, business_id, result):
    key = make_cache_key(query, business_id)

    if key in CACHE:
        CACHE.move_to_end(key)

    CACHE[key] = {
        "result": result,
        "timestamp": time.time()
    }

    if len(CACHE) > MAX_CACHE_SIZE:
        removed_key, _ = CACHE.popitem(last=False)
        print("🗑️ LRU REMOVED:", removed_key)

# CACHE_HITS = 0
# CACHE_MISSES = 0

# if data:
#     CACHE_HITS += 1
# else:
#     CACHE_MISSES += 1


def combine_multi_answers(answers):

    if not answers:
        return ""

    # remove duplicates
    answers = list(dict.fromkeys(answers))

    # binary answers
    if all(a.lower().startswith("yes") or a.lower().startswith("no") for a in answers):
        return "\n".join(answers)

    # list answers
    if any("•" in a for a in answers):
        return "\n\n".join(answers)

    # general answers
    return " ".join(answers)

def normalize_sub_query(p):

        p = p.strip()

        if any(p.startswith(w) for w in ["what","is","are","do","does","can"]):
            return p

        # 🔥 SINGLE WORD → convert
        if len(p.split()) == 1:
            return f"do you offer {p}"

        return p


class AgentExecutor:

    # def __init__(self, agent, chunk_embeddings, chunks, threshold=DEFAULT_THRESHOLD, top_k=DEFAULT_TOP_K):
    #     self.agent = agent
    #     self.pipeline = agent.pipeline
    #     self.chunk_embeddings = chunk_embeddings
    #     self.chunks = chunks
    #     self.threshold = threshold
    #     self.top_k = top_k
    # def __init__(
    #     self,
    #     agent,
    #     index,
    #     fine_chunks,
    #     coarse_chunks,
    #     fine_embeddings,
    #     coarse_embeddings,
    #     fine_bm25,
    #     coarse_bm25
    # ):
    #     self.agent = agent
    #     self.index = index
    #     self.pipeline = agent.pipeline
    #     self.fine_chunks = fine_chunks
    #     self.coarse_chunks = coarse_chunks

    #     self.fine_embeddings = fine_embeddings
    #     self.coarse_embeddings = coarse_embeddings

    #     self.fine_bm25 = fine_bm25
    #     self.coarse_bm25 = coarse_bm25
    def __init__(self, agent, index):

        self.agent = agent
        self.index = index
        self.pipeline = agent.pipeline


        fine_chunks = self.index["fine_chunks"]
        coarse_chunks = self.index["coarse_chunks"]
        list_chunks = self.index.get("list_chunks", [])

        fine_embeddings = self.index["fine_embeddings"]
        coarse_embeddings = self.index["coarse_embeddings"]

        fine_bm25 = self.index["fine_bm25"]
        coarse_bm25 = self.index["coarse_bm25"]


    def run(self, query: str, business_id: str, memory=None):

        print("🚨 EXECUTOR START")
        force_binary = False

        # -----------------------------
        # 🔥 CACHE CHECK (TOP)
        # -----------------------------
        cache_key = normalize_query(query)

        cached = get_cached_response(cache_key, business_id)

        if cached:
            print("⚡ CACHE HIT")
            return cached

        # --------------------------------------------------
        # MEMORY INITIALIZATION
        # --------------------------------------------------
        if memory is None:
            memory = AgentMemory(query)

        memory.execution["original_query"] = query
        memory.config["business_id"] = business_id
        memory.log("business_id", business_id)

        start = time.time()

        # --------------------------------------------------
        # LOAD BUSINESS CONFIG
        # --------------------------------------------------
        config = get_business_config(business_id)

        memory.config["business_name"] = config["name"]
        memory.config["tone_prompt"] = config["tone_prompt"]

        memory.config["system_prompt"] = f"""
        You are assistant for {config['name']}.
        {config['tone_prompt']}
        
        Respond in a warm, helpful and professional hospitality tone.
        Answer only from provided context.
        If information is not present, say you don't know.
        Keep answers concise and clear.
        """

        memory.log("business", config["name"])

        # --------------------------------------------------
        # STEP 1 — QUERY REWRITE
        # --------------------------------------------------
        rewritten = self.pipeline.rewrite_query(query, memory)

        # sub_queries = self.pipeline.split_multi_query(rewritten)

        normalized = normalize_query(query)
        sub_queries = self.pipeline.split_multi_query(normalized)

        sub_queries = [normalize_sub_query(p) for p in sub_queries]

        # optional (after split)
        sub_queries = [self.pipeline.rewrite_query(p, memory) for p in sub_queries]

        # 🔥 multi-query override
        if len(sub_queries) > 1:
            force_binary = True

        memory.execution["rewritten_query"] = rewritten
        memory.log("rewrite", rewritten)


        intent = detect_intent(rewritten)
        memory.execution["intent"] = intent
        memory.log("intent", intent)

        # -----------------------------
        # 🔥 QUERY TYPE DETECTION (MOVE HERE)
        # -----------------------------
        if force_binary:
            query_type = "binary"

        if is_binary_question(query):
            query_type = "binary"

        elif is_time_query(query):
            query_type = "time"

        elif is_list_query(query):
            query_type = "list"

        elif is_feature_query(query):
            query_type = "feature"   # 🔥 NEW


        else:
            query_type = "general"


        # if query_type == "general" and max_score > 10:
        #     print("🔥 INTENT OVERRIDE → LIST")
        #     query_type = "list"

        print("🧠 EXECUTOR QUERY TYPE:", query_type)


        # -----------------------------
        # 🔥 CACHE CHECK
        # # -----------------------------
        # print("🔍 CHECKING CACHE")

        # if query_type in ["time", "list", "binary", "feature"]:
        #     cache_key = normalize_query(query)
        #     cached = get_cached_response(cache_key, business_id)

        #     if cached:
        #         print("⚡ CACHE HIT")
        #         return cached


        # --------------------------------------------------
        # STEP 2 — RETRIEVE (ALWAYS EXECUTE)
        # --------------------------------------------------
        t1 = time.time()

        answers = []
        all_sources = []
        all_retrieved_chunks = []

        for q in sub_queries:

            # retrieval = retrieve_chunks(
            #     q,
            #     self.pipeline,
            #     self.chunk_embeddings,
            #     self.chunks,
            #     self.top_k,
            #     self.threshold,
            #     business_id,
            #     memory
            # )
            # retrieval = retrieve_chunks(
            #     query,
            #     self.index,
            #     self.pipeline,
            #     self.fine_chunks,
            #     self.coarse_chunks,
            #     self.fine_embeddings,
            #     self.coarse_embeddings,
            #     self.fine_bm25,
            #     self.coarse_bm25,
            #     business_id,
            #     memory
            # )

            sub_cache_key = normalize_query(q)

            cached = get_cached_response(sub_cache_key, business_id)

            if cached:
                print("⚡ SUBQUERY CACHE HIT:", q)
                answers.append(cached["answer"])
                continue

            retrieval = retrieve_chunks(
                query,
                self.index,
                self.pipeline,
                business_id,
                memory,
                query_type    # 🔥 ADD THIS
            )

            all_retrieved_chunks.extend(retrieval["chunks"])
            all_sources.extend(retrieval["sources"])

            answer_part, metrics, performance = validate_answer(
                q,
                retrieval,
                self.pipeline,
                start,
                memory.config["system_prompt"],
                memory,
                query_type = query_type
            )

            if answer_part:
                if isinstance(answer_part, dict):
                    answers.append(str(answer_part.get("answer", "")))
                else:
                    answers.append(str(answer_part))  #Add a label automatically

            if not answer_part:
                print("⚠️ EMPTY ANSWER FROM VALIDATE")
                answer_part = "I found some related information, but not enough to answer confidently."

            

        
        # combine answers from all sub-queries
        #final_answer = "\n\n".join(answers)
        final_answer = combine_multi_answers(answers)

        clean_sources = []

        for s in all_sources:

            if isinstance(s, dict):
                clean_sources.append(s.get("source", ""))
            else:
                clean_sources.append(s)

        retrieval = {
            "chunks": all_retrieved_chunks,
            "used_chunks": sum(1 for c in all_retrieved_chunks if c.get("used", True)),
            "retrieved_chunks": len(all_retrieved_chunks),
            "sources": list(set(clean_sources))
        }

        print("Retrieval time:", time.time() - t1)
        print("Sub queries:", sub_queries)

        print("\n🔍 Final retrieved chunks:", retrieval["used_chunks"])
        print("🧠 GENERATING SUGGESTIONS FOR:", query_type)

        print("DEBUG QUERY:", query)
        print("DEBUG RETRIEVAL TYPE:", type(retrieval))

        print("❌ FALLBACK: is_query_answerable")
        print("❌ FALLBACK: price_check")
        print("❌ FALLBACK: grounding")
        print("❌ FALLBACK: validate_context")
        # --------------------------------------------------
        # STEP 3 — PLANNER DECISION
        # --------------------------------------------------
        decision = plan(memory)
        memory.log("decision", decision)

        # --------------------------------------------------
        # CLARIFY
        # --------------------------------------------------
        print("✅ RETURNING RESULT")
        if decision == "CLARIFY":
            memory.log("clarify", "Asking user clarification")
            return clarify(memory, retrieval, start)

        # --------------------------------------------------
        # REFUSE
        # --------------------------------------------------
        print("✅ RETURNING RESULT")
        if decision == "REFUSE":
            return refuse(memory, retrieval, start)

        # --------------------------------------------------
        # OPTIONAL SUMMARIZATION
        # --------------------------------------------------
        if decision == "SUMMARIZE":

            retrieval = summarize_chunks(
                memory,
                retrieval,
                self.pipeline
            )

        # --------------------------------------------------
        # STEP 4 — ANSWER
        # --------------------------------------------------
        t2 = time.time()

        # answer, metrics, performance = validate_answer(
        #     query,
        #     retrieval,
        #     self.pipeline,
        #     start,
        #     memory.config["system_prompt"],
        #     memory
        # )

        print("LLM time:", time.time() - t2)

        # --------------------------------------------------
        # UPDATE CONVERSATION MEMORY
        # --------------------------------------------------
        memory.conversation["last_topic"] = rewritten
        memory.conversation["last_answer"] = answers

        first_turn = memory.conversation["last_answer"] is None

        suggestions = generate_suggestions(
            retrieval = retrieval,
            query=query,
            business_id=business_id,
            first_turn=first_turn
        )

        memory.log("answer", "Generated grounded response")
        memory.log("query_type", query_type)

        # 🔥 normalize final_answer
        meta = {}

        if isinstance(final_answer, tuple):
            answer_text, meta = final_answer
        else:
            answer_text = final_answer

        result = {
            "decision": "ANSWER",
            "answer": answer_text,
            "meta": meta, # NEW FIELD
            "suggestions": suggestions,
            "sources": retrieval["sources"],
            "retrieval": retrieval,
            "metrics": metrics,
            "performance": performance,
            "agent_trace": memory.trace
        }

        

        # -----------------------------
        # 🔥 CACHE STORE
        # -----------------------------
        print("✅ RETURNING RESULT")
        # if query_type in ["time", "list", "binary"]:
        #     set_cached_response(query, business_id, result)

        #     return result

        

        # -----------------------------
        # 🔥 STORE SUB-QUERY CACHE
        # -----------------------------
        if query_type in ["time", "list", "binary", "feature"]:
            sub_result = {
                "answer": answer_part,
                "sources": retrieval.get("sources", [])
            }

            set_cached_response(q, business_id, sub_result)

        # -----------------------------
        # 🔥 CACHE STORE (OPTIONAL)
        # # -----------------------------
        if query_type in ["time", "list", "binary", "feature"]:
            set_cached_response(cache_key, business_id, result)

        # -----------------------------
        # ✅ ALWAYS RETURN RESULT
        # -----------------------------
        print("✅ FINAL RETURN")
        print("🧠 CACHE SIZE:", len(CACHE))
        return result
        
        

        
        
        # print("❌ NO RETURN HIT")
        # print("FINAL ANSWER:", answer_text)
        # # -----------------------------
        # # 🔥 FINAL SAFE RETURN (MANDATORY)
        # # -----------------------------
        # print("⚠️ FALLBACK FINAL RETURN TRIGGERED")

        # return {
        #     "decision": "ANSWER",
        #     "answer": answer_text if 'final_answer' in locals() else "I couldn't find enough information.",
        #     "suggestions": [],
        #     "sources": retrieval.get("sources", []) if retrieval else [],
        #     "retrieval": retrieval if retrieval else {},
        #     "metrics": metrics if 'metrics' in locals() else {},
        #     "performance": performance if 'performance' in locals() else {},
        #     "agent_trace": memory.trace if memory else []
        # }


        