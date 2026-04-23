def is_time_query(question):

    q = question.lower()

    TIME_KEYWORDS = [
        "time", "timing", "hour", "hours",
        "open", "close", "opening", "closing",
        "consultation", "availability",
        "check in", "check out"
    ]

    # 🔥 strong keyword match
    if any(word in q for word in TIME_KEYWORDS):
        return True

    # 🔥 fallback patterns
    patterns = [
        "what time",
        "when does",
        "when is"
    ]

    return any(p in q for p in patterns)


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

def is_list_query(query):
    q = query.lower().strip()

    # 🔥 explicit intent
    explicit = [
        "list", "show", "give", "what are",
        "options", "available", "items"
    ]

    # 🔥 category words (domain knowledge)
    categories = [
        "facilities", "services", "amenities",
        "menu", "food", "breakfast",
        "room service", "meeting", "parking",
        "wifi", "internet", "packages"
    ]

    if any(k in q for k in explicit):
        return True

    # 🔥 implicit short queries (CRITICAL)
    if len(q.split()) <= 3:
        if any(k in q for k in categories):
            return True

    return False




def is_binary_question(question):

    q = question.lower().strip()

    patterns = ["is", "are", "do", "does", "can", "will"]

    if any(q.startswith(p + " ") for p in patterns):
        return True

    # 🔥 IMPORTANT ADDITIONS
    if "can i" in q or "is it" in q or "do you" in q:
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

def route_query(query):

    q = query.lower()

    routing = {
        "dining": ["breakfast", "restaurant", "dining", "food"],
        "parking": ["parking", "car park", "vehicle"],
        "internet": ["wifi", "internet"],
        "checkin": ["check-in", "arrival"],
        "checkout": ["check-out", "departure"],
        "facility": ["facility", "amenities", "services"]
    }

    for category, words in routing.items():
        for w in words:
            if w in q:
                return category

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


def detect_query_type(query, original_query=None, force_binary=False, list_chunks=None):

    q = query.lower().strip()
    oq = (original_query or query).lower().strip()

    # 🔥 1. binary
    if force_binary or is_binary_question(q):
        return "binary"

    # 🔥 2. time
    if is_time_query(q):
        return "time"
    
    if is_contact_query(oq):
        return "contact"

    # 🔥 3. explicit list
    if is_list_query(oq):
        return "list"

    # 🔥 4. 🔥 CRITICAL FIX (NEW)
    if list_chunks:
        for block in list_chunks:
            title = block.get("list_title", "").lower()

            # exact or partial match
            if title in oq or oq in title:
                print("🔥 LIST DETECTED FROM TITLE MATCH")
                return "list"

    # 🔥 5. feature
    if is_feature_query(q):
        return "feature"

    # 🔥 6. entity
    if len(oq.split()) <= 3:
        return "entity"

    return "general"