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
from agent.utils.query_classifier import is_time_query, detect_time_intent, is_list_query, is_binary_question, is_feature_query, detect_query_type, is_contact_query
# from rag_core import normalize_sub_query

from agent.utils.normalizer import normalize_local_query
#from agent.utils.action_generator import generate_actions
from agent.utils.HOTEL_CONFIGURATION import HOTEL_CONFIG
from agent.utils.fallback_handler import fallback_response
from agent.utils.speech_normalizer import normalize_speech_query
#from business.hotel_config import HOTEL_CONFIG

from business.hotel.actions import generate_hotel_actions

from business.hotel.intents import is_hotel_action_query
from agent.utils.query_classifier import get_fallback_response

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
from agent.utils.translator import is_hindi, to_english, to_hindi

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


def check_binary_from_list(query, list_chunks):
    q = query.lower()

    best_score = 0
    best_block = None

    for block in list_chunks:
        title = block.get("list_title", "").lower()
        items = " ".join(block.get("items", [])).lower()

        score = 0

        # 🔥 STRONG: full phrase match
        if title in q:
            score += 5

        # 🔥 word overlap with title
        overlap = sum(1 for w in q.split() if w in title)
        score += overlap * 2

        # 🔥 weaker: item match
        if any(w in items for w in q.split()):
            score += 1

        if score > best_score:
            best_score = score
            best_block = block

    # 🔥 threshold (VERY IMPORTANT)
    if best_score >= 3:
        return True, best_block

    return False, None

def binary_from_chunks(query, chunks):
    q = query.lower()

    for c in chunks:
        text = c.get("text", "").lower()

        if any(word in text for word in q.split()):
            return True, text

    return False, None



# def is_action_query(query):

#     q = query.lower().strip()

#     # request verbs
#     request_words = [
#         "need",
#         "send",
#         "bring",
#         "help",
#         "request",
#         "provide"
#     ]

#     # service nouns
#     services = [
#         "towel",
#         "blanket",
#         "wifi",
#         "food",
#         "water",
#         "cleaning",
#         "housekeeping"
#     ]

#     has_request = any(w in q for w in request_words)
#     has_service = any(s in q for s in services)

#     return has_request and has_service

def is_action_query(query):

    q = query.lower().strip()

    # --------------------------------
    # explicit request verbs
    # --------------------------------
    request_words = [
        "need",
        "send",
        "bring",
        "help",
        "request",
        "provide",
        "want"
    ]

    # --------------------------------
    # operational services
    # --------------------------------
    operational_services = [

        # room items
        "towel",
        "blanket",
        "pillow",
        "water",
        "water bottle",

        # maintenance
        "cleaning",
        "housekeeping",
        "ac",
        "air conditioning",
        "tv",

        # internet
        "wifi",
        "internet",

        # food
        "food",
        "breakfast",
        "lunch",
        "dinner",
        "tea",
        "coffee",

        # assistance
        "reception",
        "staff"
    ]

    # --------------------------------
    # informational patterns
    # --------------------------------
    # informational_patterns = [
    #     "available",
    #     "timing",
    #     "time",
    #     "what",
    #     "how",
    #     "price",
    #     "facility"
    # ]

    ACTION_VERBS = [

        "need",
        "send",
        "bring",
        "provide",
        "deliver",
        "arrange",
        "book",
        "call",

        "want",
        "require",

        "clean",
        "repair",
        "fix",

        "help me",
        "please send",
        "please bring",
        "please provide"
    ]

    has_request = any(w in q for w in request_words)
    has_service = any(s in q for s in operational_services)
    #is_informational = any(p in q for p in informational_patterns)
    has_action_verb = any(verb in q for verb in ACTION_VERBS)

    # --------------------------------
    # decision
    # --------------------------------

    # explicit service request
    if has_request and has_service:
        return True

    # # short operational query
    # if has_service and len(q.split()) <= 3 and not is_informational:
    #     return True
    if has_action_verb and has_service:
        return True

    return False


# HINGLISH_MAP = {
#     "subidha": "suvidha",
#     "suvidha": "facilities",
#     "facility": "facilities",
#     "wifi": "wifi",
#     "net": "internet",
#     "khana": "food",
#     "nashta": "breakfast"
# }

# def normalize_local_query(query):

#     q = query.lower()

#     for k, v in HINGLISH_MAP.items():
#         if k in q:
#             q = q.replace(k, v)

#     return q





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


    def run(self, query: str, business_id: str, client_id: str, memory=None):
        print("🔥🔥🔥 EXECUTOR FILE HIT 🔥🔥🔥")

        print("🚨 EXECUTOR START")

        client_id = client_id.lower()
        business_id = business_id.lower()

        
        actions = []
        original_query = query

        with open("logs/user_queries.txt", "a", encoding="utf-8") as f:
            f.write(original_query + "\n")

        # if business_id == "hotel":
        #     if is_hotel_action_query(query):
        #         print("⚡ ACTION QUERY DETECTED")

        #         return {
        #             "answer": "✅ Your request has been prepared. Reception or housekeeping will assist you shortly.",
        #             "actions": generate_hotel_actions(query, "action"),
        #             "contact": {
        #                 "phone": HOTEL_CONFIG[client_id]["phone"],
        #                 "whatsapp": HOTEL_CONFIG[client_id]["whatsapp"]
        #             },
        #             "skip_retrieval": True
        #         }

        # 🔥 NEW STEP
        query = normalize_local_query(query)
        print("🌐 original_query", original_query)
        is_hindi_query = is_hindi(original_query)
        print("🌐 is hindi :", is_hindi_query)

        if is_hindi_query:
            print("🌐 HINDI QUERY DETECTED")
            query = to_english(query)
            print("🔄 TRANSLATED QUERY:", query)

        translated_query = query  # after to_english

        metrics = {}
        performance = {}
        answer = ""
        answer_part = ""
        force_binary = False
        original_user_query = original_query

        # -----------------------------
        # 🔥 CACHE CHECK (TOP)
        # -----------------------------
        cache_key = normalize_query(query)

        # cached = get_cached_response(cache_key, business_id)
        cached = None

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

        memory.config["business_name"] = config.get("name", business_id)
        memory.config["tone_prompt"] = config.get("tone_prompt", business_id)

        memory.config["system_prompt"] = f"""
        You are assistant for {config.get("name", business_id)}.
        {config.get("tone_prompt", business_id)}
        
        Respond in a warm, helpful and professional hospitality tone.
        Answer only from provided context.
        If information is not present, say you don't know.
        Keep answers concise and clear.
        """

        memory.log("business", config.get("name", business_id))

        # --------------------------------------------------
        # STEP 1 — QUERY REWRITE
        # --------------------------------------------------
        rewritten = self.pipeline.rewrite_query(query, memory)

        # sub_queries = self.pipeline.split_multi_query(rewritten)

        # normalized = normalize_query(query)
        # sub_queries = self.pipeline.split_multi_query(normalized)

        # sub_queries = [normalize_sub_query(p) for p in sub_queries]

        # # optional (after split)
        # sub_queries = [self.pipeline.rewrite_query(p, memory) for p in sub_queries]

        # 🔥 CRITICAL FIX
        #sub_queries = [original_user_query]
        sub_queries = [translated_query]

        # 🔥 multi-query override
        if len(sub_queries) > 1:
            force_binary = True

        memory.execution["rewritten_query"] = rewritten
        memory.log("rewrite", rewritten)


        intent = detect_intent(rewritten)
        memory.execution["intent"] = intent
        memory.log("intent", intent)

        

        # query_type = detect_query_type(
        #     translated_query,  # after to_english
        #     original_query=original_user_query,  # ⚠️ IMPORTANT
        #     force_binary=force_binary,
        #     list_chunks=self.index.get("list_chunks", [])
        # )
        translated_query = normalize_speech_query(translated_query)
        route = detect_query_type(
            translated_query,  # after to_english
            original_query=original_user_query,  # ⚠️ IMPORTANT
            force_binary=force_binary,
            list_chunks=self.index.get("list_chunks", [])
        )

        if route == "action":

            print("⚡ ACTION ROUTE")

            return {
                "answer":
                "✅ Your request has been prepared. Reception or housekeeping will assist you shortly.",

                "actions":
                    generate_hotel_actions(
                        translated_query,
                        "action"
                    ),

                "contact": {
                    "phone":
                        HOTEL_CONFIG[client_id]["phone"],

                    "whatsapp":
                        HOTEL_CONFIG[client_id]["whatsapp"]
                },

                "skip_retrieval": True
            }
        


        print("🧠 EXECUTOR QUERY TYPE:", route)
        print("ACTION CHECK:", query)
        print("IS ACTION:", is_hotel_action_query(query))

        intent = detect_time_intent(translated_query)

        #print("🧠 EXECUTOR QUERY TYPE:", query_type)

        




        # --------------------------------------------------
        # STEP 2 — RETRIEVE (ALWAYS EXECUTE)
        # --------------------------------------------------
        t1 = time.time()

        answers = []
        all_sources = []
        all_retrieved_chunks = []

        for q in sub_queries:

            

            sub_cache_key = normalize_query(q)

            #cached = get_cached_response(sub_cache_key, business_id)
            cached = None

            if cached:
                print("⚡ SUBQUERY CACHE HIT:", q)
                answers.append(cached["answer"])
                continue

            retrieval = retrieve_chunks(
                translated_query,
                self.index,
                self.pipeline,
                business_id,
                memory,
                route, 
                intent    # 🔥 ADD THIS
            )

            retrieval_route = route  # since we passed it

            if retrieval_route != route:
                print("⚠️ ROUTE MISMATCH:", route, retrieval_route)


            intent = retrieval.get("intent")

            if (
                intent is not None
                and not retrieval.get("evidence_found", True)
                ):
                    return {
                        "answer": get_fallback_response(intent),
                        "sources": [],
                        "actions": []
                    }
            

            all_retrieved_chunks.extend(retrieval["chunks"])
            all_sources.extend(retrieval["sources"])

            answer_part, metrics, performance = validate_answer(
                translated_query,
                retrieval,
                self.pipeline,
                start,
                memory.config["system_prompt"],
                memory,
                query_type = route
            )

            if answer_part:
                if isinstance(answer_part, dict):
                    answers.append(str(answer_part.get("answer", "")))
                else:
                    answers.append(str(answer_part))  #Add a label automatically

            if not answer_part:

                if route == "contact":
                    return "Contact information is not available.", {}, {}

                print("⚠️ EMPTY ANSWER FROM VALIDATE")
                answer_part, actions = fallback_response(q, retrieval.get("chunks"))

        
            #actions = meta.get("actions", [])

        
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
        print("🧠 GENERATING SUGGESTIONS FOR:", route)

        print("DEBUG QUERY:", query)
        print("🔍 RETRIEVAL QUERY:", query)
        print("DEBUG RETRIEVAL TYPE:", type(retrieval))

        print("❌ FALLBACK: is_query_answerable")
        print("❌ FALLBACK: price_check")
        print("❌ FALLBACK: grounding")
        print("❌ FALLBACK: validate_context")

        print("🧠 ORIGINAL QUERY:", original_user_query)
        print("🧠 IS HINDI:", is_hindi_query)
        print("🧠 FINAL QUERY:", query)
        print("🧠 SUB QUERIES:", sub_queries)
        print("🧠 FINAL QUERY TYPE:", route)

        print(is_hindi("कौन-कौन सी सुविधाएँ उपलब्ध हैं"))

        print("🧠 ORIGINAL QUERY:", original_query)
        print("🧠 IS HINDI:", is_hindi_query)
        print("BUSINESS ID:", business_id)
        print("AVAILABLE CONFIG KEYS:", HOTEL_CONFIG.keys())
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

        if is_hindi_query and suggestions:
            suggestions = [to_hindi(s) for s in suggestions]

        memory.log("answer", "Generated grounded response")
        memory.log("query_type", route)

        memory.log("route", route)

        # 🔥 normalize final_answer
        meta = {}

        if isinstance(final_answer, tuple):
            answer_text, meta = final_answer
        else:
            answer_text = final_answer

        if is_hindi_query:
            print("🔄 TRANSLATING RESPONSE TO HINDI")
            answer_text = to_hindi(answer_text)

        #actions = generate_hotel_actions(answer_text, route, original_query)
        actions = generate_hotel_actions(original_query, route)


        client_config = HOTEL_CONFIG.get(client_id, {})

        result = {
            "decision": "ANSWER",
            "answer": answer_text,
            #"actions": actions if actions else generate_hotel_actions(answer_text, route, q),
            "actions": actions if actions else generate_hotel_actions(q, route),
            "contact": {
                "phone": client_config.get("phone"),
                "whatsapp": client_config.get("whatsapp")
            },
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

        print("📥 CONTACT CHUNKS:", retrieval.get("chunks"))

        # -----------------------------
        # 🔥 STORE SUB-QUERY CACHE
        # -----------------------------
        if route in ["time", "list", "binary", "feature"]:
            sub_result = {
                "answer": answer_part,
                "sources": retrieval.get("sources", [])
            }

            set_cached_response(q, business_id, sub_result)

        # -----------------------------
        # 🔥 CACHE STORE (OPTIONAL)
        # # -----------------------------
        if route in ["time", "list", "binary", "feature"]:
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


        