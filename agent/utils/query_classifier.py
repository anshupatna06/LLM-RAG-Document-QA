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

def is_list_query(question):
    q = question.lower().strip()

    # 🔥 explicit keywords
    list_keywords = [
        "list", "show", "give", "what are", "options",
        "available", "items", "services"
    ]

    # 🔥 category keywords (VERY IMPORTANT)
    category_keywords = [
        "dishes", "menu", "food", "facilities",
        "services", "amenities", "options",
        "treatments", "packages"
    ]

    # ✅ case 1: explicit list intent
    if any(k in q for k in list_keywords):
        return True

    # ✅ case 2: pure noun query (like "popular dishes")
    words = q.split()

    if len(words) <= 3:  # short queries → likely intent-based
        if any(k in q for k in category_keywords):
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
