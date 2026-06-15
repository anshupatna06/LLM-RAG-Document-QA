# def is_time_query(question):

#     q = question.lower()

#     TIME_KEYWORDS = [
#         "time", "timing", "hour", "hours",
#         "open", "close", "opening", "closing",
#         "consultation", "availability",
#         "check in", "check out"
#     ]

#     # 🔥 strong keyword match
#     if any(word in q for word in TIME_KEYWORDS):
#         return True

#     # 🔥 fallback patterns
#     patterns = [
#         "what time",
#         "when does",
#         "when is"
#     ]

#     return any(p in q for p in patterns)

from business.hotel.intents import is_hotel_action_query

INTENT_RETRIEVAL_KEYWORDS = {

    "checkin_time": [
        "check in",
        "check-in"
    ],

    "checkout_time": [
        "check out",
        "check-out"
    ],

    "breakfast_time": [
        "breakfast"
    ],

    "room_service_time": [
        "room service"
    ],

    "wifi_password": [
        "wifi",
        "wi-fi",
        "internet"
    ],

    "parking_info": [
        "parking"
    ],

    "food_menu": [
        "menu",
        "food"
    ]
}


def has_intent_evidence(
    intent,
    retrieved_chunks
):

    keywords = (
        INTENT_RETRIEVAL_KEYWORDS.get(
            intent,
            []
        )
    )

    if not keywords:
        return True

    for score, text, source in retrieved_chunks:

        text = text.lower()

        for keyword in keywords:

            if keyword in text:
                return True

    return False


INTENT_FALLBACKS = {

    "checkin_time":
        "I could not find check-in timing information in the uploaded documents. Please contact reception for confirmation.",

    "checkout_time":
        "I could not find check-out timing information in the uploaded documents. Please contact reception for confirmation.",

    "wifi_password":
        "I could not find WiFi information in the uploaded documents. Please contact reception for assistance.",

    "parking_info":
        "I could not find parking information in the uploaded documents.",

    "food_menu":
        "I could not find food menu information in the uploaded documents."
}

def get_fallback_response(intent):

    return INTENT_FALLBACKS.get(
        intent,
        "I could not find information related to your request in the uploaded documents."
    )

TIME_INTENT_KEYWORDS = {

    "checkin_time": [
        "check in",
        "check-in",
        "early check in",
        "early check-in"
    ],

    "checkout_time": [
        "check out",
        "check-out",
        "late checkout",
        "late check out"
    ],

    "breakfast_time": [
        #"breakfast",
        "morning buffet"
    ],

    "room_service_time": [
        "room service",
        "food service"
    ],

    "restaurant_time": [
        "restaurant timing",
        "restaurant open",
        "restaurant close"
    ]
}


# # def is_time_query(q):

#     q = q.lower()

#     # 🔥 direct time intent
#     if re.search(r'(what time|when|timing|kab)', q):
#         return True

#     # 🔥 structured pattern
#     if re.search(r'(open|close|check.?in|check.?out)', q):
#         return True

#     return False

def detect_time_intent(query):

    q = query.lower()

    for intent, keywords in TIME_INTENT_KEYWORDS.items():

        for keyword in keywords:

            if keyword in q:
                return intent

    # generic timing intent
    if re.search(
        r'(what time|timing|when|kab)',
        q
    ):
        return "general_time"

    return None


def is_time_query(query):

    return detect_time_intent(query) is not None


# ----------------------------
# QUERY TYPE DETECTION
# ----------------------------
# def is_list_query(question):
#     q = question.lower()

#     list_keywords = [
#         "facilities", "services", "amenities", "features",
#         "options", "offerings"
#     ]

#     list_phrases = [
#         "what are", "which are", "list", "show", "tell me"
#     ]

#     return (
#         any(k in q for k in list_keywords) or
#         any(p in q for p in list_phrases)
#     )

# def is_list_query(question):
#     q = question.lower().strip()

#     # 🔥 explicit keywords
#     list_keywords = [
#         "list", "show", "give", "what are", "options",
#         "available", "items", "services"
#     ]

#     # 🔥 category keywords (VERY IMPORTANT)
#     category_keywords = [
#         "dishes", "menu", "food", "facilities",
#         "services", "amenities", "options",
#         "treatments", "packages"
#     ]

#     # ✅ case 1: explicit list intent
#     if any(k in q for k in list_keywords):
#         return True

#     # ✅ case 2: pure noun query (like "popular dishes")
#     words = q.split()

#     if len(words) <= 3:  # short queries → likely intent-based
#         if any(k in q for k in category_keywords):
#             return True

#     return False

# def is_list_query(query):
#     q = query.lower().strip()

#     # 🔥 explicit intent
#     explicit = [
#         "list", "show", "give", "what are",
#         "options", "available", "items"
#     ]

#     # 🔥 category words (domain knowledge)
#     categories = [
#         "facilities", "services", "amenities",
#         "menu", "food", "breakfast",
#         "room service", "meeting", "meeting room", "parking",
#         "wifi", "internet", "packages"
#     ]

#     if any(k in q for k in explicit):
#         return True

#     # 🔥 implicit short queries (CRITICAL)
#     if len(q.split()) <= 3:
#         if any(k in q for k in categories):
#             return True

#     return False

def is_list_query(q):

    q = q.lower().strip()

    # 🔥 Pattern 1: asking for "set of things"
    if re.search(r'(what (are|all)|kya kya|which|list|show)', q):
        return True

    # 🔥 Pattern 2: noun-only queries (entity → list)
    if len(q.split()) <= 2:
        return True

    # 🔥 Pattern 3: plural intent
    if re.search(r'\b(s|services|facilities|amenities)\b', q):
        return True

    return False


# def is_binary_question(question):

#     q = question.lower().strip()

#     patterns = ["is", "are", "do", "does", "can", "will"]

#     if any(q.startswith(p + " ") for p in patterns):
#         return True

#     # 🔥 IMPORTANT ADDITIONS
#     if "can i" in q or "is it" in q or "do you" in q:
#         return True

#     return False

import re

def is_binary_question(q):

    q = q.lower().strip()

    # 🔥 Pattern 1: starts with auxiliary verb
    if re.match(r'^(is|are|do|does|can|will|should)\b', q):
        return True

    # 🔥 Pattern 2: ends with question style (Hindi / Hinglish)
    if re.search(r'(hai kya|milta hai|available hai)\??$', q):
        return True

    return False


def is_feature_query(query):
    q = query.lower()

    keywords = [
        "parking", "wifi", "internet",
        "pool", "gym", "spa",
        "ac", "air conditioning",
        "lift", "elevator",
        "doctor", "pharmacy"
    ]

    return any(k in q for k in keywords)


# ----------------------------------------
# SEMANTIC ROUTING
# ----------------------------------------

# def route_query(query):

#     q = query.lower()

#     routing = {
#         "dining": ["breakfast", "restaurant", "dining", "food"],
#         "parking": ["parking", "car park", "vehicle"],
#         "internet": ["wifi", "internet"],
#         "checkin": ["check-in", "arrival"],
#         "checkout": ["check-out", "departure"],
#         "facility": ["facility", "amenities", "services"]
#     }

#     for category, words in routing.items():
#         for w in words:
#             if w in q:
#                 return category

#     return "general"


def route_query(q):

    q = q.lower()

    if re.search(r'(wifi|internet)', q):
        return "internet"

    if re.search(r'(parking|car)', q):
        return "parking"

    if re.search(r'(food|breakfast|restaurant)', q):
        return "dining"

    return "general"


# def detect_query_type(query, original_query=None, force_binary=False):

#     q = query.lower().strip()
#     oq = (original_query or query).lower().strip()

#     # 🔥 1. binary (highest priority)
#     if force_binary or is_binary_question(q):
#         return "binary"

#     # 🔥 2. time
#     if is_time_query(q):
#         return "time"

#     # 🔥 3. list (IMPORTANT — use ORIGINAL query)
#     if is_list_query(oq):
#         return "list"

#     # 🔥 4. feature (your existing logic)
#     if is_feature_query(q):
#         return "feature"

#     # 🔥 5. entity (CRITICAL FIX)
#     if len(oq.split()) <= 3:
#         return "entity"

#     return "general"


def is_contact_query(query):
    q = query.lower()

    keywords = [
        "contact", "contact number",
        "phone", "phone number", "number",
        "mobile", "call", "reach",
        "telephone"
    ]

    return any(k in q for k in keywords)


# def detect_query_type(query, original_query=None, force_binary=False, list_chunks=None):

#     q = query.lower().strip()
#     oq = (original_query or query).lower().strip()

#     # 🔥 1. binary
#     if force_binary or is_binary_question(q):
#         return "binary"

#     # 🔥 2. time
#     if is_time_query(q):
#         return "time"
    
#     if is_contact_query(oq):
#         return "contact"

#     # 🔥 3. explicit list
#     if is_list_query(oq):
#         return "list"

#     # 🔥 4. 🔥 CRITICAL FIX (NEW)
#     if list_chunks:
#         for block in list_chunks:
#             title = block.get("list_title", "").lower()

#             # exact or partial match
#             if title in oq or oq in title:
#                 print("🔥 LIST DETECTED FROM TITLE MATCH")
#                 return "list"

#     # 🔥 5. feature
#     if is_feature_query(q):
#         return "feature"

#     # 🔥 6. entity
#     if len(oq.split()) <= 3:
#         return "entity"

#     return "general"

# def detect_query_type(query, original_query=None, force_binary=False, list_chunks=None):

#     q = query.lower().strip()
#     oq = (original_query or query).lower().strip()

#     # =========================================
#     # 🔥 1. HARD RULES (DO NOT REMOVE THESE)
#     # =========================================

#     # ✅ contact (highest priority)
#     if is_contact_query(oq):
#         return "contact"

#     # ✅ time
#     if is_time_query(q):
#         return "time"
    
#     # ✅ binary (HIGH PRIORITY)
#     if is_binary_question(q):
#         return "binary"

#     # ✅ explicit list
#     if is_list_query(oq):
#         return "list"

#     # ✅ list title match (VERY STRONG SIGNAL)
#     if list_chunks:
#         for block in list_chunks:
#             title = block.get("list_title", "").lower()
#             if title and (title in oq or oq in title):
#                 print("🔥 LIST DETECTED FROM TITLE MATCH")
#                 return "list"
            
#     if any(word in q for word in ["price", "cost", "rate", "per night", "rent"]):
#         return "feature"   # or "price" if you create new type

#     # =========================================
#     # 🔥 2. SCORING (SOFT DECISION LAYER)
#     # =========================================

#     scores = {
#         "list": 0,
#         "binary": 0,
#         "feature": 0,
#         "entity": 0,
#         "general": 0
#     }

#     # -------------------------
#     # LIST SIGNALS
#     # -------------------------
#     if "facilities" in q or "services" in q:
#         scores["list"] += 2

#     if len(oq.split()) <= 3:
#         scores["entity"] += 2
#         #scores["list"] += 1   # 🔥 important bias

#     # if len(q.split()) <= 2:
#     #     return "list"

#     # -------------------------
#     # BINARY SIGNALS
#     # -------------------------
#     if force_binary or is_binary_question(q):
#         scores["binary"] += 2

#     if "hai kya" in q or "available" in q:
#         scores["binary"] += 3

#     # -------------------------
#     # FEATURE SIGNALS
#     # -------------------------
#     if is_feature_query(q):
#         scores["feature"] += 1

#     # -------------------------
#     # DEFAULT GENERAL
#     # -------------------------
#     scores["general"] += 0

#     # =========================================
#     # 🔥 3. FINAL DECISION
#     # =========================================

#     #route = max(scores, key=scores.get)
#     max_score = max(scores.values())

#     if max_score == 0:
#         print("⚠️ NO SIGNAL → FALLBACK TO GENERAL")
#         return "general"
    
#     candidates = [k for k, v in scores.items() if v == max_score]

#     # priority order
#     priority = ["binary", "list", "time", "contact", "entity", "general"]

#     for p in priority:
#         if p in candidates:
#             return p

#     route = max(scores, key=scores.get)

#     print("🧠 ROUTING SCORES:", scores)
#     print("🧠 FINAL ROUTE:", route)

#     return route

# def detect_query_type(q):

#     if is_contact_query(q):
#         return "contact"

# #     if is_time_query(q):
# #         return "time"

# #     if is_binary_question(q):
# #         return "binary"

# #     if is_list_query(q):
# #         return "list"

# #     return "general"

# def detect_query_type(query, original_query=None):

#     q = query.lower().strip()
#     oq = (original_query or query).lower().strip()

#     # =========================================
#     # 🔥 HARD RULES (PRIORITY ORDER)
#     # =========================================

#     # ✅ contact (use original language)
#     if is_contact_query(oq):
#         return "contact"

#     # ✅ time (structure-based)
#     if is_time_query(q):
#         return "time"

#     # ✅ binary (use BOTH)
#     if is_binary_question(q) or is_binary_question(oq):
#         return "binary"

#     # ✅ list (use BOTH)
#     if is_list_query(oq) or is_list_query(q):
#         return "list"

#     # =========================================
#     # 🔥 FALLBACK
#     # =========================================

#     return "general"

import re

def detect_query_type(
    query,
    original_query=None,
    force_binary=False,
    list_chunks=None
):

    q = query.lower().strip()
    oq = (original_query or query).lower().strip()

    # =========================================
    # 🔥 1. HARD OVERRIDES
    # =========================================

    if force_binary:
        print("⚡ FORCE BINARY TRIGGERED")
        return "binary"

    # =========================================
    # 🔥 2. HARD RULES (STRUCTURE)
    # =========================================

    if is_contact_query(oq):
        return "contact"

    if is_time_query(q):
        return "time"

    if is_binary_question(q) or is_binary_question(oq):
        return "binary"
    
    if is_hotel_action_query(oq):
        return "action"

    # =========================================
    # 🔥 3. LIST DETECTION (PATTERN FIRST)
    # =========================================

    if is_list_query(oq) or is_list_query(q):
        return "list"

    # =========================================
    # 🔥 4. SEMANTIC BOOST (SECONDARY SIGNAL)
    # =========================================

    if list_chunks:
        for block in list_chunks:
            title = block.get("list_title", "").lower()

            if title and (title in oq or title in q):
                print("🔥 LIST BOOST FROM TITLE MATCH")
                return "list"

    # =========================================
    # 🔥 5. FALLBACK
    # =========================================

    return "general"