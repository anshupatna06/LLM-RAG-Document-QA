from agent.utils.intent import detect_intent
import re
from agent.tools.validate import is_time_query

import math
from agent.utils.query_classifier import (
    is_time_query,
    is_list_query,
    is_binary_question,
    is_feature_query,
    route_query,
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


# ----------------------------------------
# ADAPTIVE TOP-K
# ----------------------------------------

def dynamic_top_k(query, default_k):

    q = query.lower()

    if is_time_query(q):
        return 5

    if is_list_query(q):
        return 6

    if is_binary_question(q):
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


# ----------------------------------------
# MAIN RETRIEVAL PIPELINE
# ----------------------------------------

def retrieve_chunks(query, index, pipeline, business_id, memory, query_type):
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

    # -----------------------------
    # NORMALIZE QUERY
    # -----------------------------
    query = normalize_query(query)

    

    query_type = detect_query_type(
        query,
        original_query=original_user_query,
        list_chunks=index.get("list_chunks", [])
    )
    print("RETRIEVAL QUERY TYPE:", query_type)


    # -----------------------------
    # # 🔥 CONTACT EARLY RETURN
    # # -----------------------------
    # if query_type == "contact":

    #     print("🚀 CONTACT EARLY RETURN TRIGGERED")

    #     contact_blocks = [
    #         c for c in list_chunks
    #         if c.get("type") == "contact"
    #     ]

    #     if contact_blocks:
    #         block = contact_blocks[0]

    #         return {
    #             "chunks": [
    #                 {
    #                     "text": block["text"],
    #                     "source": block,
    #                     "type": "contact",
    #                     "score": 1.0
    #                 }
    #             ],
    #             "used_chunks": 1,
    #             "retrieved_chunks": 1,
    #             "sources": [block["source"]]
    #         }
    
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
                "sources": [top[0]]
            }



    # -----------------------------
    # 🔥 STEP 3: NORMAL RETRIEVAL SETUP
    # # -----------------------------
    # if query_type == "time":
    #     # 🔥 time lives in structured / coarse chunks
    #     active_chunks = coarse_chunks
    #     active_embeddings = coarse_embeddings
    #     active_bm25 = coarse_bm25

    # elif query_type == "list":
    #     # 🔥 list handled separately (early return below)
    #     active_chunks = coarse_chunks
    #     active_embeddings = coarse_embeddings
    #     active_bm25 = coarse_bm25

    # elif query_type == "binary":
    #     active_chunks = coarse_chunks
    #     active_embeddings = coarse_embeddings
    #     active_bm25 = coarse_bm25

    # elif query_type == "feature":
    #     active_chunks = coarse_chunks
    #     active_embeddings = coarse_embeddings
    #     active_bm25 = coarse_bm25

    # else:  # general
    #     active_chunks = coarse_chunks
    #     active_embeddings = coarse_embeddings
    #     active_bm25 = coarse_bm25

    active_chunks = coarse_chunks
    active_embeddings = coarse_embeddings
    active_bm25 = coarse_bm25



    # -----------------------------
    # 🔥 LIST QUERY ROUTING (NEW)
    # -----------------------------
    # if is_list_query(query):

    #     list_chunks = index.get("list_chunks", [])

    #     if list_chunks:
    #         ranked = sorted(
    #             list_chunks,
    #             key=lambda c: score_list_block(query, c),
    #             reverse=True
    #         )

            

    #         print("\n🏆 RANKED LIST CHUNKS:")
    #         for r in ranked[:5]:
    #             print(r["list_title"])

    #         # return TOP MATCH directly
    #         top = ranked[:1]
            

    #         print("🔥 ENTERED LIST PIPELINE")

    #         list_chunks = index.get("list_chunks", [])

    #         print("📊 AVAILABLE LIST CHUNKS:", len(list_chunks))

    #         print("\n✅ RETURNING LIST CHUNKS:")
    #         for t in top:
    #             print(t["list_title"], t["items"])

    #         return {
    #             "chunks": top,
    #             "used_chunks": len(top),
    #             "retrieved_chunks": len(top),
    #             "sources": top
    #         }

    
        

     


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
            "sources": []
        }

    # -----------------------------
    # ADAPTIVE TOP-K
    # -----------------------------
    adaptive_k = dynamic_top_k(query, BASE_TOP_K)

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
            adaptive_k
        )
    else:
        vector_results = []

    # -----------------------------
    # BM25 RETRIEVAL
    # -----------------------------
    keyword_results = []

    if active_bm25:
        raw_results = active_bm25.search(query, k=adaptive_k)

        for c in raw_results:

            if isinstance(c, tuple):
                score, text, source = c
            else:
                score = c.get("score", 1.0)   # default score
                text = c.get("text", "")
                source = c.get("source", "")

            score = float(score) if isinstance(score, (int, float, str)) else 0.0

            if isinstance(source, dict):
                if source.get("business_id") == business_id:
                    keyword_results.append({
                        "score": score,
                        "text": text,
                        "source": source
                    })

    
    # -----------------------------
    # HYBRID MERGE (FIXED ✅)
    # -----------------------------
    combined = {}

    # -----------------------------
    # STEP 1: ADD VECTOR RESULTS
    # -----------------------------
    for c in vector_results:

        if isinstance(c, tuple):
            score, text, source = c
        else:
            score = c.get("score", 1.0)   # default score
            text = c.get("text", "")
            source = c.get("source", "")

        
        if isinstance(source, dict):
            chunk_id = source.get("chunk_id")
        else:
            chunk_id = hash(text)   # fallback (temporary)

        combined[chunk_id] = {
            "text": text,
            "vector": score,
            "bm25": 0,
            "source": source
        }

    
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0


    # 🔥 normalize scores FIRST
    normalized_results = [
        (safe_float(score), text, source)
        for score, text, source in keyword_results
    ]

    # 🔥 compute max safely
    max_bm25 = max([s for s, _, _ in normalized_results], default=1.0)

    if max_bm25 == 0:
        max_bm25 = 1.0   # avoid division by zero

    # main loop
    for score, text, source in normalized_results:
        if isinstance(source, dict):
            chunk_id = source.get("chunk_id")
        else:
            chunk_id = hash(text)   # fallback (temporary)
            
        #bm25_scaled = score / max_bm25
        if max_bm25 == 0:
            bm25_scaled = 0.0
        else:
            bm25_scaled = score / max_bm25

        if chunk_id not in combined:
            combined[chunk_id] = {
                "text": text,
                "vector": 0.0,
                "bm25": bm25_scaled,
                "source": source
            }
        else:
            combined[chunk_id]["bm25"] = bm25_scaled

    is_time = is_time_query(query)
    retrieved = []

    for chunk_id, scores in combined.items():

        hybrid_score = (
            scores["vector"] * 0.6 +
            scores["bm25"] * 0.4
        )

        
        if is_time_query(query):
            if re.search(r'\d{1,2}:\d{2}|closed', scores["text"], re.I):
                hybrid_score += 1.0   # 🔥 VERY STRONG BOOST

        
        if is_time_query(query):
            if "to" in scores["text"].lower():
                hybrid_score += 0.5

        if is_list_query(query):
            if re.search(r'\d{1,2}:\d{2}', scores["text"]):
                hybrid_score += 1.0

        target_section = map_query_to_section(query)

        source = scores["source"]

        if isinstance(source, dict):
            section = source.get("section", "")
        else:
            section = ""   # fallback

        if target_section and target_section in section:
            hybrid_score += 0.3


        hybrid_score = safe_score(hybrid_score)

        retrieved.append(
            (hybrid_score, scores["text"], scores["source"])
        )



    retrieved.sort(reverse=True)

    # 🔥 FORCE include time chunks for time queries
    time_chunks = [] #ALWAYS NITIALIZE FIRST
    if is_time_query(query):

        
        time_chunks = [
            r for r in retrieved
            if re.search(r'\d{1,2}:\d{2}|closed', r[1], re.I)
        ]

        if time_chunks:
            # 🔥 KEEP ALL TIME CHUNKS (NO DEDUP LOSS)
            retrieved = sorted(time_chunks, reverse=True)
            # seen = set()
            # final = []

            # for item in time_chunks + retrieved:
            #     if item[1] not in seen:
            #         final.append(item)
            #         seen.add(item[1])

            # retrieved = final

#     def normalize_to_tuple(c):
#     if isinstance(c, tuple):
#         return c[:3]
#     elif isinstance(c, dict):
#         return (c.get("score", 0), c.get("text", ""), c.get("source", ""))
#     return None

# results = [normalize_to_tuple(c) for c in results if c]

    retrieved = [normalize_chunk(c) for c in retrieved if c]


    # 🔥 ADD THIS BLOCK

    list_chunks = index.get("list_chunks", [])

    if list_chunks:
        for block in list_chunks:
            for item in block.get("items", []):

                retrieved.append((0.9, item, block))

    # 🔥 boost for binary/feature
    if query_type in ["binary", "feature"]:
        print("🔥 BOOSTING LIST ITEMS")

        boosted = []

        for c in retrieved:

            if isinstance(c, tuple):
                if len(c) >= 3:
                    score, text, source = c[:3]
                else:
                    continue

            elif isinstance(c, dict):
                score = c.get("score", 0)
                text = c.get("text", "")
                source = c.get("source", "")
            else:
                continue

            # 🔥 boost logic
            if isinstance(source, dict) and source.get("list_title"):
                score += 1.0

            boosted.append((score, text, source))

        retrieved = boosted

    # -----------------------------
    # RERANK
    # -----------------------------
    retrieved = rerank(query, retrieved)
    

    
    # -----------------------------
    # DEBUG LOGS (UNCHANGED ✅)
    # -----------------------------
    print("Total chunks available:", len(coarse_chunks))
    print("Filtered chunks:", len(filtered_chunks))
    print("Vector results:", len(vector_results))
    print("BM25 results:", len(keyword_results))
    print("Hybrid results:", len(retrieved))

    

    for c in retrieved:

        if isinstance(c, tuple):
            score, text, source = c
        else:
            score = c.get("score", 1.0)   # default score
            text = c.get("text", "")
            source = c.get("source", "")

        print(f"{score:.3f}", text[:100])

    print("\n🔥 AFTER HYBRID MERGE:")


    for c in retrieved:

        if isinstance(c, tuple):
            score, text, source = c
        else:
            score = c.get("score", 1.0)   # default score
            text = c.get("text", "")
            source = c.get("source", "")

        print(score, text[:80])

    print("\n🔥 FINAL USED CHUNKS:")

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

    elif is_binary_question(query):
        threshold = 0.25  # stricter

    else:
        threshold = 0.3   # default

    for idx, c in enumerate(retrieved, 1):

        if isinstance(c, tuple):
            score, text, source = c
        else:
            score = c.get("score", 1.0)   # default score
            text = c.get("text", "")
            source = c.get("source", "")

        #score = float(score)
        max_score = max(max_score, score)

        # if max_score < 0.3:
        #     return LOW_CONFIDENCE_RESPONSE

        # always take top 2
        is_used = (
            idx <= TOP_K_USED
            or score >= threshold
        )
        
        if is_used:
            used.append(text)

        debug.append({
            "rank": idx,
            "score": score,
            "text": text,
            "source": source,
            "used": is_used
        })
        if max_score < 0.15:
            memory.log("low_confidence", True)


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

    print("QUERY:", query)
    print("IS TIME:", is_time_query(query))
    print("IS LIST:", is_list_query(query))

    

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
        ]))
    }