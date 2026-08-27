from agent.utils.intent import detect_intent
import re

from agent.tools.chunk_utils import normalize_chunks
from agent.tools.retrieval_source_selection import select_retrieval_index
from agent.tools.scoring import calculate_final_score
import math
from agent.utils.query_classifier import (
    is_time_query,
    detect_time_intent,
    INTENT_EVIDENCE_REGISTRY,
    has_intent_evidence,
    get_fallback_response,
    is_list_query,
    is_binary_query,
    is_feature_query,
    route_query,
    calculate_domain_entity_bonus,
    detect_query_type
)

def safe_score(x):
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return float(x)



# ----------------------------------------
# RERANK
# ----------------------------------------




def rerank(query, results):

    query_words = set(query.lower().split())

    boosted_keywords = [
        "facility", "facilities", "service", "services",
        "wifi", "parking", "breakfast", "access", "room"
    ]

    scored = []

    for c in results:

        # 🔥 HANDLE BOTH FORMATS
        if isinstance(c, tuple):
            score, text, source = c[:3]

        elif isinstance(c, dict):
            score = c.get("score", 1.0)
            text = c.get("text", "")
            source = c.get("source", "")

        else:
            continue

        t = text.lower()

        overlap = len(query_words & set(t.split()))

        boost = 0

        # ✅ semantic boost
        for word in boosted_keywords:
            if word in t:
                boost += 0.3

        # ✅ short informative boost
        if len(t.split()) <= 12:
            boost += 0.2

        final_score = score + (overlap * 0.1) + boost

        scored.append((final_score, text, source))

    scored.sort(key = lambda x : x[0], reverse=True)

    return scored




def rerank_by_intent(
    chunks,
    intent
):

    keywords = (
        INTENT_EVIDENCE_REGISTRY.get(
            intent,
            []
        )
    )

    if not keywords:
        return chunks

    scored = []

    for chunk in chunks:

        score = 0

        text = chunk.lower()

        for keyword in keywords:

            if keyword in text:
                score += 10

        scored.append(
            (score, chunk)
        )

    scored.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return [
        chunk
        for score, chunk
        in scored
    ]




# ----------------------------------------
# ADAPTIVE TOP-K
# ----------------------------------------

def dynamic_top_k(query, default_k):

    q = query.lower()

    if is_time_query(q):
        return 5

    if is_list_query(q):
        return 6

    if is_binary_query(q):
        return 3

    if len(q.split()) <= 3:
        return 2

    if len(q.split()) <= 6:
        return 3

    return default_k



# ----------------------------------------
# QUERY NORMALIZATION
# ----------------------------------------

def normalize_query(query):

    q = query.lower().strip()

    replacements = {
        "?": "",
        "included in": "included",
        "available in": "available",
        "complimentary": "included",
        "free breakfast": "breakfast included",
        "car park": "parking",
        "wifi": "internet wifi"
    }

    for k, v in replacements.items():
        q = q.replace(k, v)

    return q


# ----------------------------------------
# INTENT BOOSTING
# ----------------------------------------

def boost_query(query):

    hotel_intents = {
        "breakfast": ["breakfast", "restaurant", "dining", "food"],
        "parking": ["parking", "car park", "vehicle", "garage"],
        "checkin": ["check-in", "check in", "arrival"],
        "checkout": ["check-out", "departure"],
        "wifi": ["wifi", "internet"],
        "pool": ["pool", "swimming"]
    }

    boost_terms = []

    for key, words in hotel_intents.items():
        if key in query:
            boost_terms.extend(words)

    if boost_terms:
        query = query + " " + " ".join(boost_terms)

    return query



def map_query_to_section(query):

    q = query.lower()

    # 🔥 DIRECT KEYWORD MAPPING (STRONG SIGNAL)
    if any(w in q for w in ["hour", "time", "open", "close", "timing"]):
        return "hours"

    if any(w in q for w in ["facility", "amenities", "services"]):
        return "services"

    if any(w in q for w in ["food", "menu", "restaurant"]):
        return "dining"

    # 🔥 FALLBACK: intent-based
    intent = detect_intent(query)

    intent_map = {
        "time": "hours",
        "facility": "services",
        "food": "dining"
    }

    return intent_map.get(intent, None)


def resolve_target_section(
    query,
    entities,
    index
):

    entity_section_map = index.get(
        "entity_section_map",
        {}
    )

    for entity in entities:

        section = entity_section_map.get(entity)

        if section:
            return section

    # Legacy/general fallback
    return map_query_to_section(query)


# ----------------------------------------
# BUSINESS + ROUTE FILTERING
# ----------------------------------------

# def filter_chunks(chunks, embeddings, business_id, query):

#     #target_section = map_query_to_section(query)

#     filtered_chunks = []
#     filtered_embeddings = []

#     for emb, chunk in zip(embeddings, chunks):

#         # business filter
#         if chunk["business_id"] != business_id:      # ✅ ONLY strict filter = business
#             continue

#         # section filter (soft)
#         # if target_section and target_section not in chunk.get("section", ""):
#         #     continue

#         filtered_chunks.append(chunk)
#         filtered_embeddings.append(emb)

    

#     return filtered_chunks, filtered_embeddings

def filter_chunks(chunks, embeddings, business_id, query):

    filtered_chunks = []
    filtered_embeddings = []

    # 🔥 CASE 1: embeddings NOT available
    if embeddings is None:
        for chunk in chunks:
            if chunk["business_id"] != business_id:
                continue
            filtered_chunks.append(chunk)

        return filtered_chunks, None

    # 🔥 CASE 2: embeddings available
    for emb, chunk in zip(embeddings, chunks):
        if chunk["business_id"] != business_id:
            continue

        filtered_chunks.append(chunk)
        filtered_embeddings.append(emb)

    return filtered_chunks, filtered_embeddings

# NORMALIZE SCORE TP PREVENT DOMINATION OF BM25 OVER EMBEDDING [EMBEDDING: 0--> 1  , BM25: 0--> 10+  , SO BM25 ALWAYS WINS]
def normalize_scores(results):

    if not results:
        return results

    scores = [score for score, _, _ in results]

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return results

    normalized = []

    for c in results:

        score = c.get("score", 1.0)   # default score
        text = c.get("text", "")
        source = c.get("source", "")

        norm = (score - min_s) / (max_s - min_s)

        normalized.append((norm, text, source))

    return normalized


def normalize(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower())


# def score_list_block(query, block):

#     q = normalize(query)
#     title = normalize(block.get("list_title", ""))

#     score = 0

#     q_words = set(q.split())
#     title_words = set(title.split())

#     # ✅ 1. EXACT MATCH (STRONGEST)
#     if title in q:
#         score += 15

#     # ✅ 2. TOKEN OVERLAP (CORE SIGNAL)
#     overlap = len(q_words & title_words)
#     score += overlap * 4

#     # ✅ 3. PARTIAL MATCH (WEAK)
#     if any(word in title for word in q_words):
#         score += 2

#     # ✅ 4. LIGHT INTENT BOOST (NOT DOMINANT)
#     intent_map = {
#         "dish": ["dish", "food", "menu"],
#         "facility": ["facility", "amenity"],
#         "service": ["service"],
#         "hour": ["hour", "timing", "opening"]
#     }

#     for key, words in intent_map.items():
#         if key in q:
#             if any(w in title for w in words):
#                 score += 2   # small boost only

#     return score

def score_list_block(query, block):

    q = query.lower()

    title = block.get("list_title", "").lower()
    items = " ".join(block.get("items", [])).lower()

    score = 0

    # ----------------------------
    # 🔥 DIRECT MATCH (TITLE)
    # ----------------------------
    if any(word in title for word in q.split()):
        score += 2

    # ----------------------------
    # 🔥 DIRECT MATCH (ITEMS) ← MOST IMPORTANT
    # ----------------------------
    if any(word in items for word in q.split()):
        score += 5   # 🔥 stronger than title

    # ----------------------------
    # 🔥 SYNONYM MATCH
    # ----------------------------
    SYNONYMS = {
        "health": ["health", "checkup", "preventive", "Preventive health checkups"],
        "checkup": ["checkup", "checkups", "preventive", "Preventive health checkups"],
        "diagnostic": ["diagnostic", "lab", "test"],
        "consultation": ["consultation", "doctor", "appointment"],
    }

    for key, syns in SYNONYMS.items():
        if key in q:
            if any(s in items for s in syns):
                score += 6   # 🔥 strongest boost

    return score



def normalize_chunk(c):

    if isinstance(c, tuple):
        score, text, source = c
        return {
            "score": float(score),
            "text": text,
            "source": source,
            "used": True
        }

    elif isinstance(c, dict):
        return {
            "score": float(c.get("score", 0)),
            "text": c.get("text", ""),
            "source": c.get("source", ""),
            "used": c.get("used", True)
        }

    return None

def debug_retrieval_stage(label, chunks):

    print("\n" + "=" * 80)
    print(f"🔎 {label}")
    print("=" * 80)

    for rank, c in enumerate(chunks, 1):

        if isinstance(c, dict):

            print(
                f"{rank:02d} | "
                f"score={c.get('final_score', 'NO_FINAL_SCORE')} | "
                f"text={c.get('text', '')[:80]}"
            )

        elif isinstance(c, tuple):

            print(
                f"{rank:02d} | "
                f"TUPLE | "
                f"score={c[0] if len(c) > 0 else 'N/A'} | "
                f"text={c[1][:80] if len(c) > 1 else ''}"
            )

        else:

            print(
                f"{rank:02d} | "
                f"UNKNOWN TYPE={type(c)} | "
                f"{c}"
            )

    print("=" * 80)


def debug_bm25_query(bm25, query, targets, k=20):

    print("\n" + "=" * 80)
    print("🔬 BM25 QUERY DEBUG")
    print("=" * 80)

    print("QUERY:", repr(query))
    print("K:", k)
    print("INDEX SIZE:", len(bm25.chunks))

    results = bm25.search(query, k=k)

    print("\nTOP RESULTS:")

    for rank, result in enumerate(results, 1):

        print(
            f"{rank:02d} | "
            f"{float(result['score']):.6f} | "
            f"{result['text']}"
        )

    print("\nTARGET RECALL:")

    for target in targets:

        found = any(
            target.lower() in result["text"].lower()
            for result in results
        )

        print(
            f"{target!r} → {found}"
        )


# ----------------------------------------
# MAIN RETRIEVAL PIPELINE
# ----------------------------------------

def retrieve_chunks(query, index, pipeline, business_id, memory, query_type, intent = None, entities = None):
    fine_chunks = index["fine_chunks"]
    coarse_chunks = index["coarse_chunks"]
    list_chunks = index.get("list_chunks", [])

    fine_embeddings = index["fine_embeddings"]
    coarse_embeddings = index["coarse_embeddings"]

    fine_bm25 = index["fine_bm25"]
    coarse_bm25 = index["coarse_bm25"]
    
    BASE_TOP_K = {
        "time": 5,
        "list": 6,
        "binary": 3,
        "default": 3
    }

    original_user_query = query
    translated_query = query

    # -----------------------------
    # NORMALIZE QUERY
    # -----------------------------
    query = normalize_query(query)

    

    # query_type = detect_query_type(
    #     translated_query,  # after to_english
    #     original_query=original_user_query,
    #     list_chunks=index.get("list_chunks", [])
    # )
    print("RETRIEVAL QUERY TYPE:", query_type)


    
    
    # -----------------------------
    # 🔥 HADLE LIST EARLY
    # -----------------------------
    if query_type == "list":

        list_chunks = index.get("list_chunks", [])

        if list_chunks:
            ranked = sorted(
                list_chunks,
                key=lambda c: score_list_block(original_user_query, c),
                reverse=True
            )

            top = ranked[:1]

            # 🔥 DO NOT RETURN
            # 🔥 INSTEAD CONVERT INTO STANDARD FORMAT

            return {
                "chunks": [
                    {
                        "text": " | ".join(top[0]["items"]),
                        "items": top[0]["items"],
                        "score": 1.0,
                        "source": top[0],
                        "type": "list"
                    }
                ],
                "used_chunks": 1,
                "retrieved_chunks": 1,
                "sources": [top[0]],
                "evidence_found": True,
                "intent": None
            }



    
    # active_chunks = coarse_chunks
    # active_embeddings = coarse_embeddings
    # active_bm25 = coarse_bm25


    active_chunks, active_embeddings, active_bm25 = \
        select_retrieval_index(
            query_type,
            index
        )


    print("\n" + "=" * 80)
    print("🔒 QUERY BEFORE BM25 DEBUG")
    print("=" * 80)
    print("Production query:", repr(query))
    print("Query type:", query_type)
    print("Entities:", entities)
    print("Active chunks:", len(active_chunks))
    print("Active BM25 chunks:", len(active_bm25.chunks))

    debug_bm25_query(
        active_bm25,
        "breakfast time",
        ["Complimentary breakfast (7:00 AM - 10:00 AM)"],
        k=20
    )

    # print("\n========== ACTIVE BM25 ==========")


    # for i, chunk in enumerate(active_bm25.chunks):
    #     if (
    #         "laundry" in chunk["text"].lower()
    #         or "swimming pool" in chunk["text"].lower()
    #         or "room service" in chunk["text"].lower()
    #     ):
    #         print(
    #             i,
    #             repr(chunk["text"]),
    #             "section=",
    #             chunk.get("section")
    #         )

    # for debug_query in [
    #     "laundry",
    #     "swimming pool",
    #     "room service"
    # ]:
    #     print("\nQUERY:", debug_query)

    #     results = active_bm25.search(
    #         debug_query,
    #         k=5
    #     )

    #     for r in results:
    #         print(
    #             r["score"],
    #             "|",
    #             r["text"]
    #         )

    # print("\n" + "=" * 80)
    # print("🔒 QUERY AFTER BM25 DEBUG")
    # print("=" * 80)
    # print("Production query:", repr(query))
    
        
    # print("\n" + "=" * 80)
    # print("🎯 ACTUAL BM25 SEARCH")
    # print("=" * 80)

    # print("original query:", query)
    # print("retrieval query:", query)
    # print("entities:", entities)

    # results = active_bm25.search(query, k=20)

    # print("\nRESULTS:")
    # for i, r in enumerate(results, 1):
    #     print(i, r["score"], repr(r["text"]))
     


    # -----------------------------
    # FILTER BY BUSINESS
    # -----------------------------
    filtered_chunks, filtered_embeddings = filter_chunks(
        active_chunks,
        active_embeddings,
        business_id,
        query
    )

    if not filtered_chunks:
        return {
            "chunks": [],
            "used_chunks": 0,
            "retrieved_chunks": 0,
            "sources": [],

            "evidence_found": False,
            "intent": None
        }

    # -----------------------------
    # ADAPTIVE TOP-K
    # -----------------------------
    adaptive_k = dynamic_top_k(query, BASE_TOP_K)
    retrieval_k = max(adaptive_k, 20)
    # -----------------------------
    # VECTOR RETRIEVAL
    # -----------------------------
    # vector_results = pipeline.retrieve(
    #     query,
    #     filtered_embeddings,
    #     filtered_chunks,
    #     adaptive_k
    # )

    # -----------------------------
    # VECTOR RETRIEVAL (ONLY IF AVAILABLE)
    # -----------------------------
    if filtered_embeddings is not None:
        vector_results = pipeline.retrieve(
            query,
            filtered_embeddings,
            filtered_chunks,
            retrieval_k
        )
    else:
        vector_results = []

    # -----------------------------
    # BM25 RETRIEVAL
    # -----------------------------
    keyword_results = []

    # if active_bm25:

    #     print("\n" + "=" * 70)
    #     print("🔬 BM25 CANDIDATE RECALL TEST")
    #     print("=" * 70)
    #     print("Query:", query)

    #     test_ks = [3, 5, 10, 20]

    #     for k in test_ks:

    #         test_results = active_bm25.search(
    #             query,
    #             k=k
    #         )

    #         found = False

    #         for result in test_results:
  
    #             if isinstance(result, tuple):
    #                 score, text, source = result
    #             else:
    #                 score = result.get("score", 0)
    #                 text = result.get("text", "")
    #                 source = result.get("source", {})

    #             if "laundry" in text.lower():
    #                 found = True

    #                 print(
    #                     f"  K={k} → FOUND: "
    #                     f"{text[:100]}"
    #                 )

    #                 break

    #         # print(
    #         #     f"K = {k:<2} | "
    #         #     f"Laundry found = {found}"
    #         # )
    #         print(f"\nK = {k}")
    #         print("-" * 50)

    #         found = any(
    #             "laundry" in (
    #             r[1] if isinstance(r, tuple)
    #             else r.get("text", "")
    #             ).lower()
    #             for r in test_results
    #         )

    #         print(f"K={k} → Laundry found: {found}")

    #         for rank, result in enumerate(test_results, 1):

    #             if isinstance(result, tuple):
    #                 score, text, source = result
    #             else:
    #                 score = result.get("score", 0)
    #                 text = result.get("text", "")

    #             print(
    #                 f"{rank}. "
    #                 f"{float(score):.3f} | "
    #                 f"{text[:80]}"
    #             )

    #             #print(f"🎯 Laundry found: {found}")

    #     print("=" * 70)

    if active_bm25:
        raw_results = active_bm25.search(query, k=retrieval_k)

    #     print("===== RAW BM25 =====")
    #     for r in raw_results:
    #         print(r)
    #     print("="*60)
    # # # #     # for c in raw_results:

    # # #     #     if isinstance(c, tuple):
    # # #     #         score, text, source = c
    # # #     #     else:
    # # #     #         score = c.get("score", 1.0)   # default score
    # # #     #         text = c.get("text", "")
    # # #     #         source = c.get("source", "")

    # # #     #     score = float(score) if isinstance(score, (int, float, str)) else 0.0

    # # #     #     if isinstance(source, dict):
    # # #     #         if source.get("business_id") == business_id:
    # # #     #             keyword_results.append({
    # # #     #                 "score": score,
    # # #     #                 "text": text,
    # # #     #                 "source": source
    # # #     #             })

        for result in raw_results:

            source = result["source"]

            if source["business_id"] == business_id:
 
                keyword_results.append(result)

    
    # -----------------------------
    # HYBRID MERGE (FIXED ✅)
    # -----------------------------
    combined = {}

    # -----------------------------
    # STEP 1: ADD VECTOR RESULTS
    # # -----------------------------
    

    for result in vector_results:

        chunk = normalize_chunks(
            result,
            "vector"
        )

        combined[
            chunk["chunk_id"]
        ] = chunk

    
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0


    # normalized_results = []

    # for result in keyword_results:

    #     normalized_results.append(
    #         (
    #             safe_float(result["score"]),
    #             result["text"],
    #             result["source"]
    #         )
    #     )

    # 🔥 compute max safely
    #max_bm25 = max([s for s, _, _ in keyword_results], default=1.0)

    # if max_bm25 == 0:
    #     max_bm25 = 1.0   # avoid division by zero

    max_bm25 = max(
        [
            safe_float(result["score"])
            for result in keyword_results
        ],
        default=1.0 
    )


    for result in keyword_results:

        chunk = normalize_chunks(result, "bm25")

        chunk["bm25_score"] /= max_bm25

        if chunk["chunk_id"] not in combined:
 
            combined[chunk["chunk_id"]] = chunk

        else:

            combined[chunk["chunk_id"]]["bm25_score"] = chunk["bm25_score"]

    is_time = is_time_query(query)
    # intent = detect_time_intent(query)


    keywords = (
        INTENT_EVIDENCE_REGISTRY.get(
            intent,
            []
        )
    )
    retrieved = []

    SCORING_CONFIG = {

        "intent_bonus": 2.0,

        "section_bonus": 0.3,

        "list_bonus": 1.0,

        "pattern_bonus": 1.0,

        "vector_weight": 0.6,

        "bm25_weight": 0.4,

        "entity_bonus": 2.0

    }


    target_section = resolve_target_section(query, entities, index)
    print("🚥🚥🚥TARGET SECTION DETECTED IN RETRIEVAL PIPELINE:", target_section)

    query_context = {

        "query": query,

        "route": query_type,

        "intent": intent,

        "entities": entities,

        "target_section": target_section,

        "is_list": is_list_query(query),

        "is_time": is_time_query(query),
    }

    # for chunk_id, chunk in combined.items():

    #     print("CHUNK ID:", chunk_id)
    #     print(chunk)

    #     retrieval_score = (
    #         chunk["vector_score"] * SCORING_CONFIG["vector_weight"] +
    #         chunk["bm25_score"] * SCORING_CONFIG["bm25_weight"]
    #     )

    #     intent_bonus = 0.0
    #     section_bonus = 0.0
    #     list_bonus = 0.0
    #     pattern_bonus = 0.0

        
    #     # if is_time_query(query):
    #     #     if re.search(r'\d{1,2}:\d{2}|closed', scores["text"], re.I):
    #     #         hybrid_score += 1.0   # 🔥 VERY STRONG BOOST

        
    #     # if is_time_query(query):
    #     #     if "to" in scores["text"].lower():
    #     #         hybrid_score += 0.5
    #     print("=" * 60)
    #     print("SCORES DICT:")
    #     print(chunk)
    #     text = chunk["text"].lower()

    #     for keyword in keywords:
    #         if keyword in text:
    #             #hybrid_score += 2.0
    #             #intent_bonus += SCORING_CONFIG["intent_bonus"]
    #             config = INTENT_EVIDENCE_REGISTRY[intent]

    #             intent_bonus += config["bonus"]
    #             print(f"Intent    : {intent_bonus:.2f}")

    #     print("ENTITIES INSIDE RETRIEVE:", entities)
    #     entity_bonus = calculate_domain_entity_bonus(text, entities)

    #     if is_list_query(query):
    #         if re.search(r'\d{1,2}:\d{2}', chunk["text"]):
    #             #hybrid_score += 1.0
    #             list_bonus += SCORING_CONFIG["list_bonus"]
                

        

    #     section = chunk["source"]

    #     # if isinstance(source, dict):
    #     #     section = source.get("section", "")
    #     # else:
    #     #     section = ""   # fallback

    #     if target_section and target_section in section:
    #         #hybrid_score += 0.3
    #         section_bonus += SCORING_CONFIG["section_bonus"]


    #     #hybrid_score = safe_score(hybrid_score)


    #     final_score = (
    #         retrieval_score +
    #         intent_bonus +
    #         entity_bonus +
    #         section_bonus +
    #         pattern_bonus +
    #         list_bonus
    #     )

    #     chunk["final_score"] = final_score

    #     # retrieved.append(
    #     #     (final_score, scores["text"], scores["source"])
    #     # )
    #     retrieved.append(chunk)


        # print("=" * 60)
        # print(chunk["text"][:70])
        # print(f"Retrieval : {retrieval_score:.2f}")
        # print(f"Intent    : {intent_bonus:.2f}")
        # print(f"Entity    : {entity_bonus:.2f}")
        # print(f"Section   : {section_bonus:.2f}")
        # print(f"Pattern   : {pattern_bonus:.2f}")
        # print(f"List      : {list_bonus:.2f}")
        # print(f"Final     : {final_score:.2f}")
        # print("=" * 60)


    for chunk_id, chunk in combined.items():
        

        chunk["final_score"] = calculate_final_score(
            chunk,
            query_context
        )

        retrieved.append(chunk)
        # print("=" * 60)
        # print(f"Final     : {chunk['final_score']:.2f}")
        # print("=" * 60)
        # if (
        #     "laundry" in chunk["text"].lower()
        #     or "room service" in chunk["text"].lower()
        # ):

        #     print("\n" + "=" * 70)
        #     print("🎯 TARGET SCORING DEBUG")
        #     print("TEXT:", chunk["text"])
        #     print("BM25:", chunk["bm25_score"])
        #     print("VECTOR:", chunk["vector_score"])

        #     print("FINAL:", chunk["final_score"])
        #     print("=" * 70)

    debug_retrieval_stage(
        "AFTER FEATURE SCORING",
        retrieved
    )

    
    print("INTENT:", intent)

    #retrieved.sort(reverse=True)
    #TypeError: '<' not supported between instances of 'dict' and 'dict'

    retrieved.sort(
        key=lambda chunk: chunk["final_score"],
        reverse=True
    )

    debug_retrieval_stage(
        "AFTER FINAL_SCORE SORT",
        retrieved
    )


    # 🔥 FORCE include time chunks for time queries
    
    # time_chunks = [] #ALWAYS NITIALIZE FIRST
    # if is_time_query(query):

        
    #     time_chunks = [
    #         r for r in retrieved
    #         if re.search(r'\d{1,2}:\d{2}|closed', r[1], re.I)
    #     ]

    #     if time_chunks:
    #         # 🔥 KEEP ALL TIME CHUNKS (NO DEDUP LOSS)
    #         retrieved = sorted(time_chunks, reverse=True)
            

    time_chunks = []

    if is_time_query(query):

        for r in retrieved:

            if not isinstance(r, dict):
                continue

            text = r.get("text", "")

            if re.search(
                r'\d{1,2}:\d{2}|closed',
                text,
                re.I
            ):
                time_chunks.append(r)

        if time_chunks:

            retrieved = sorted(
                time_chunks,
                key=lambda x: x.get("final_score", 0.0), 
                reverse=True
            )

    debug_retrieval_stage(
        "BEFORE normalize_chunk",
        retrieved
    )

    # retrieved = [normalize_chunk(c) for c in retrieved if c]

    # debug_retrieval_stage(
    #     "AFTER normalize_chunk",
    #     retrieved
    # )

    debug_retrieval_stage(
        "AFTER SCORE SORT — CANONICAL CHUNKS",
        retrieved
    )


    # # 🔥 ADD THIS BLOCK
    

    # list_chunks = index.get("list_chunks", [])

    # if list_chunks:
    #     for block in list_chunks:
    #         for item in block.get("items", []):

    #             retrieved.append((0.9, item, block))


    # # 🔥 boost for binary/feature
    # if query_type in ["binary", "feature"]:
    #     print("🔥 BOOSTING LIST ITEMS")

    #     boosted = []

    #     for c in retrieved:

    #         if isinstance(c, tuple):
    #             if len(c) >= 3:
    #                 score, text, source = c[:3]
    #             else:
    #                 continue

    #         elif isinstance(c, dict):
    #             score = c.get("score", 0)
    #             text = c.get("text", "")
    #             source = c.get("source", "")
    #         else:
    #             continue

    #         # 🔥 boost logic
    #         if isinstance(source, dict) and source.get("list_title"):
    #             score += 1.0

    #         boosted.append((score, text, source))

    #     retrieved = boosted

    # -----------------------------
    # RERANK
    # -----------------------------

    debug_retrieval_stage(
        "BEFORE RERANK",
        retrieved
    )
    
    # retrieved = rerank(query, retrieved)

    debug_retrieval_stage(
        "AFTER RERANK",
        retrieved
    )    
    

    
    # -----------------------------
    # DEBUG LOGS (UNCHANGED ✅)
    # -----------------------------
    print("Total chunks available:", len(coarse_chunks))
    print("Filtered chunks:", len(filtered_chunks))
    print("Vector results:", len(vector_results))
    print("BM25 results:", len(keyword_results))
    print("Hybrid results:", len(retrieved))
    print("INTENT =", detect_time_intent(query))

    

    print("\n🔥 CURRENT RANKING:")

    for rank, chunk in enumerate(retrieved, 1):

        print(
            f"{rank}. "
            f"{chunk['final_score']:.3f} | "
            f"{chunk['text'][:100]}"
        )


    print("\n🔥 FINAL FEATURE-SCORED RANKING:")

    for rank, chunk in enumerate(retrieved, 1):

        print(
            f"{rank}. "
            f"{chunk['final_score']:.3f} | "
            f"{chunk['text'][:100]}"
        )


    print("SCORE TYPE:", type(chunk.get("score")))
    print("FINAL SCORE TYPE:", type(chunk.get("final_score")))
    print("USED TYPE:", type(chunk.get("used")))
    


    # for c in retrieved:

    #     if isinstance(c, tuple):
    #         score, text, source = c
    #     else:
    #         score = c.get("score", 1.0)   # default score
    #         text = c.get("text", "")
    #         source = c.get("source", "")

    #     print(t[:100])

    print("\n🧠 FINAL SCORING:")
    for c in list_chunks:
        print(c["list_title"], "→", score_list_block(original_user_query, c))

    evidence_found = True
    if intent != "general_time":
        evidence_found = has_intent_evidence(intent, retrieved)

    print("FINAL INTENT:", intent)
    print("RETRIEVED SAMPLE:", retrieved[:3])
    print("FINAL EVIDENCE:", evidence_found)


    # for score, text, source in retrieved[:5]:
    #     print(score, text[:100])



    # -----------------------------
    # THRESHOLD FILTER
    # -----------------------------
    used = []
    debug = []
    max_score = 0.0
    TOP_K_USED = 5 if is_time_query(query) else 2

    # -----------------------------
    # ADAPTIVE THRESHOLD
    # -----------------------------
    if is_time_query(query):
        threshold = 0.2   # allow more recall

    elif is_list_query(query):
        threshold = 0.15  # broader retrieval

    elif is_feature_query(query):
        threshold = 0.20 

    elif is_binary_query(query):
        threshold = 0.25  # stricter

    else:
        threshold = 0.3   # default


    print("\n" + "=" * 80)
    print("🎯 THRESHOLD / SELECTION DEBUG")
    print("=" * 80)

    print("Query:", query)
    print("Query type:", query_type)
    print("TOP_K_USED:", TOP_K_USED)
    print("Threshold:", threshold)
    print("Retrieved count:", len(retrieved))

    for rank, chunk in enumerate(retrieved, 1):

        print(
            f"{rank}. "
            f"final_score={chunk['final_score']:.3f} | "
            f"text={chunk['text'][:100]}"
        )

    print("=" * 80)

    # for idx, c in enumerate(retrieved, 1):

    #     if isinstance(c, tuple):
    #         score, text, source = c
    #     else:
    #         score = c.get("score", 1.0)   # default score
    #         text = c.get("text", "")
    #         source = c.get("source", "")

    #     #score = float(score)
    #     max_score = max(max_score, score)

    #     # if max_score < 0.3:
    #     #     return LOW_CONFIDENCE_RESPONSE

    #     # always take top 2
    #     is_used = (
    #         idx <= TOP_K_USED
    #         or score >= threshold
    #     )
        
    #     if is_used:
    #         used.append(text)

    #     debug.append({
    #         "rank": idx,
    #         "score": score,
    #         "text": text,
    #         "source": source,
    #         "used": is_used
    #     })
    #     if max_score < 0.15:
    #         memory.log("low_confidence", True)

    for idx, chunk in enumerate(retrieved, 1):

        score = float(chunk["final_score"])
        text = chunk["text"]
        source = chunk["source"]

        max_score = max(max_score, score)

        is_used = (
            idx <= TOP_K_USED
            or score >= threshold
        )

        print(
            f"\nRANK {idx}"
        )

        print("TEXT:", text)
        print("FINAL SCORE:", score)
        print("THRESHOLD:", threshold)
        print("TOP-K CONDITION:", idx <= TOP_K_USED)
        print("THRESHOLD CONDITION:", score >= threshold)
        print("USED:", is_used)

        if is_used:
            used.append(text)

        debug.append({
            "rank": idx,
            "score": score,
            "final_score": score,
            "text": text,
            "source": source,
            "used": is_used
        })

    print("\n" + "=" * 80)
    print("🧠 SELECTED EVIDENCE")
    print("=" * 80)

    for i, text in enumerate(used, 1):
        print(f"{i}. {text}")

    print("TOTAL USED:", len(used))
    print("=" * 80)


    # -----------------------------
    # FALLBACK (IMPORTANT)
    # -----------------------------
    if len(used) == 0 and len(debug) > 0:

        best = max(debug, key=lambda x: x["score"])
        best["used"] = True
        used.append(best["text"])

    # -----------------------------
    # MEMORY UPDATE
    # -----------------------------
    memory.execution["max_score"] = max_score
    memory.execution["used_chunks"] = len(used)

    memory.log("retrieve", f"used={len(used)} max_score={max_score:.3f}")
    memory.log("adaptive_top_k", adaptive_k)
    #memory.log("route", route)

    # -----------------------------
    # FINAL DEBUG
    # -----------------------------
    print("Used chunks:", len(used))
    print("Query:", query)
    #print("Route:", route)
    print("Adaptive K:", adaptive_k)

    #print("Chunk business ids:", [c["business_id"] for c in chunks])
    print("Query business id:", business_id)

    print("\n🚦 QUERY TYPE CHECK")
    print("Query:", query)
    print("Is List:", is_list_query(query))
    print("Is Binary:", is_binary_query(query))

    print("QUERY:", query)
    print("IS TIME:", is_time_query(query))
    print("IS LIST:", is_list_query(query))




    print("\n" + "=" * 80)
    print("🚨 RETRIEVAL RETURN DEBUG")
    print("=" * 80)

    print("USED CHUNKS:")
    for i, text in enumerate(used, 1):
        print(i, text)

    print("DEBUG CHUNKS:")
    for item in debug:
        print(
            item["rank"],
            item["final_score"],
            item["used"],
            item["text"]
        )

    print("=" * 80)
    
    # -----------------------------
    # RETURN
    # -----------------------------
    return {
        "chunks": debug,
        "used_chunks": len(used),
        "retrieved_chunks": len(debug),
        "sources": list(set([
           c["source"]["source"] if isinstance(c["source"], dict) else c["source"]
           for c in debug if c.get("used", True)
        ])),
        "evidence_found": evidence_found,
        "intent": intent
    }