import re


# def matches_route(query, route_config):

#     q = query.lower().strip()

#     for pattern in route_config["patterns"]:

#         if re.search(pattern, q):
#             return True

#     return False

import re

def matches_route(query, route_config):

    q = query.lower().strip()

    rules = route_config.get("rules", [])

    for rule in rules:

        pattern = rule["pattern"]
        rule_type = rule.get("type", "search")

        if rule_type == "search":
            if re.search(pattern, q):
                return True

        elif rule_type == "match":
            if re.match(pattern, q):
                return True

    return False


             


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


INTENT_EVIDENCE_REGISTRY = {

    "checkin_time": {
        "keywords": [
            "check in",
            "check-in",
            "arrival",
            "checkin"
        ],

        "bonus" : 2.0
    },

    "checkout_time": {
        "keywords":[
            "check out",
            "check-out",
            "departure",
            "checkout"
        ],
        "bonus": 2.0
    },

    "breakfast_time": {
        "keywords": [
            "breakfast",
            "buffet",
            "morning buffet",
            "breakfast timing"
        ],
        "bonus":2.0
    },

    "room_service_time": {
        "keywords": [
            "room service",
            "food service",
            "in-room dining"
        ],
        "bonus":2.0
    },

    "wifi_password": {
        "keywords":
        [
            "wifi",
            "wi-fi",
            "internet",
            "wireless",
            "network"
        ],
        "bonus":2.0
    },

    "parking_info": {
        "keywords":[
            "parking",
            "car parking",
            "vehicle parking",
            "parking area"
        ],
        "bonus":2.0
    },

    "food_menu": {
        "keywords":[
            "menu",
            "food",
            "restaurant menu",
            "dishes",
            "meal"
        ],
        "bonus":2.0
    },

    "laundry_service": {
        "keywords":[
            "laundry",
            "dry cleaning",
            "washing",
            "clothes wash"
        ],
        "bonus":2.0
    },

    "swimming_pool": {
            "keywords":[
            "swimming pool",
            "pool",
            "indoor pool"
        ],
        "bonus":2.0
    },

    "gym_facility": {
        "keywords":[
            "gym",
            "fitness",
            "fitness center",
            "workout"
        ],
        "bonus":2.0
    },

    "spa_service": {
        "keywords":[
            "spa",
            "massage",
            "wellness"
        ],
        "bonus":2.0
    },

    "housekeeping": {
        "keywords":[
            "housekeeping",
            "cleaning",
            "room cleaning"
        ],
        "bonus":2.0
    },

    "airport_pickup": {
        "keywords":[
            "airport pickup",
            "airport drop",
            "pickup",
            "drop"
        ],
        "bonus":2.0
    },

    "conference_room": {
        "keywords":[
            "conference",
            "meeting room",
            "conference hall"
        ],
        "bonus":2.0
    },

    "banquet_hall": {
        "keywords":[
            "banquet",
            "banquet hall",
            "event hall",
            "wedding hall"
        ],
        "bonus":2.0
    },

    "coffee_shop": {
        "keywords":[
            "coffee shop",
            "cafe",
            "bakery",
            "coffee"
        ],
        "bonus":2.0
    }
}

# def has_intent_evidence(
#     intent,
#     retrieved_chunks
# ):

#     keywords = (
#         INTENT_EVIDENCE_REGISTRY.get(
#             intent,
#             []
#         )
#     )

#     if not keywords:
#         return True

#     for score, text, source in retrieved_chunks:

#         text = text.lower()

#         for keyword in keywords:

#             if keyword in text:
#                 return True

#     return False

def has_intent_evidence(
    intent,
    retrieved_chunks
):

    config = INTENT_EVIDENCE_REGISTRY.get(intent)

    if not config:
        return True

    keywords = config.get("keywords", [])

    if not keywords:
        return True

    for chunk in retrieved_chunks:

        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text", "").lower()

        for keyword in keywords:

            if keyword.lower() in text:
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

TIME_INTENTS = {

    "checkin_time": {
        "keywords": [
            "check in",
            "check-in",
            "early check in",
            "early check-in"
        ],
        "route": "time",
        "priority": 10,
        "businesses": [
            "hotel"
        ],

        "validation" : [
            "checkin"
        ]
    },

    "checkout_time": {

        "keywords": [
        "check out",
        "check-out",
        "late checkout",
        "late check out"
        ],
        
        "route": "time",
        "priority": 10,
        "businesses": [
            "hotel"
        ],

        "validation":[
            "checkout",
            "departure"
        ]
    },

    "breakfast_time": {
        "keywords": [
        "breakfast",
        "morning buffet",
        ],
        
        "route": "time",
        "priority": 10,
        "businesses": [
            "hotel"
        ],

        "validation":[
            "breakfast"
        ]
    },

    "room_service_time": {
        "keywords": [
            "room service",
            "food service"
        ],

        "route": "time",
        "priority": 10,
        "businesses": [
            "hotel"
        ],

        "validation":[
            "room service"
        ]
    },

    "restaurant_time": {
        "keywords": [
        "restaurant timing",
        "restaurant open",
        "restaurant close"
        ],
        
        "route": "time",
        "priority": 10,
        "businesses": [
            "hotel"
        ],

        "validation":[
            "restaurant time"
        ]
    }
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

    intent = detect_intent(
        query,
        TIME_INTENTS
    )

    if intent:
        return intent

    # generic timing intent
    if re.search(
        r'(what time|timing|when|kab)',
        query
    ):
        return "general_time"

    return None

TIME_ROUTE = {

    "rules":[

        {
            "type":"search",
            "pattern":r"\bwhat time\b"
        },

        {
            "type":"search",
            "pattern":r"\btime\b"
        },

        {
            "type":"search",
            "pattern":r"\bwhen\b"
        },

        {
            "type":"search",
            "pattern":r"\bhours\b"
        },

        {
            "type":"search",
            "pattern":r"\bopen\b"
        },

        {
            "type":"search",
            "pattern":r"\bclose\b"
        },
        
        {
            "type":"search",
            "pattern":r"\bkab\b"
        },
        
        {
            "type":"search",
            "pattern":r"\bkis samay\b"
        },

        {
            "type":"search",
            "pattern":r"\btimings?\b"
        }

    ]

}

# TIME_ROUTE = {

#     "patterns": [

#         r"\bwhat time\b",
#         r"\bwhen\b",
#         r"\btimings?\b",
#         r"\bhours?\b",
#         r"\bopen\b",
#         r"\bclose\b",
#         r"\bkab\b"

#     ]
# }

# def is_time_query(query):

#     q = query.lower().strip()

#     for pattern in TIME_ROUTE["patterns"]:

#         if re.search(pattern, q):
#             return True

#     return False

def is_time_query(query):

    return matches_route(
        query,
        TIME_ROUTE
    )

# def is_time_query(query):

#     return detect_time_intent(query) is not None


# ----------------------------
# QUERY TYPE DETECTION
# ----------------------------

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

LIST_ROUTE = {

    "rules":[

        {
            "type":"search",
            "pattern":r"\blist\b"
        },

        {
            "type":"search",
            "pattern":r"\bshow\b"
        },

        {
            "type":"search",
            "pattern":r"\bshow me\b"
        },

        {
            "type":"search",
            "pattern":r"\bfacilities\b"
        },

        {
            "type":"search",
            "pattern":r"\bwhat are\b"
        },

        {
            "type":"search",
            "pattern":r"\bwhat all\b"
        },
        
        {
            "type":"search",
            "pattern":r"\bwhich\b"
        },
        
        {
            "type":"search",
            "pattern":r"\bavailable options\b"
        },

        {
            "type":"search",
            "pattern":r"\bkya kya\b"
        }

    ]

}
# LIST_ROUTE = {

#     "patterns":[

#         "list",
#         "show",
#         "show me",
#         "what are",
#         "what all",
#         "which",
#         "available options",
#         "kya kya"

#     ]
# }

# def is_list_query(query):

#     q = query.lower().strip()

#     return any(
#         pattern in q
#         for pattern in LIST_ROUTE["patterns"]
#     )

def is_list_query(query):

    return matches_route(
        query,
        LIST_ROUTE
    )


LIST_INTENTS = {

    "facilities":{

        "keywords":[
            "facilities",
            "facility",
            "amenities"
        ],

        "route":"list"

    },

    "services":{

        "keywords":[
            "services",
            "service"
        ],

        "route":"list"

    },

    "room_types":{

        "keywords":[
            "rooms",
            "room types"
        ],

        "route":"list"

    },

    "restaurant_menu":{

        "keywords":[
            "menu",
            "food menu"
        ],

        "route":"list"

    },

    "nearby_attractions":{

        "keywords":[
            "nearby",
            "places",
            "attractions"
        ],

        "route":"list"

    }

}

def detect_list_intent(query):

    return detect_intent(
        query,
        LIST_INTENTS
    )

# def is_list_query(q):

#     q = q.lower().strip()

#     # 🔥 Pattern 1: asking for "set of things"
#     if re.search(r'(what (are|all)|kya kya|which|list|show)', q):
#         return True

#     # 🔥 Pattern 2: noun-only queries (entity → list)
#     if len(q.split()) <= 2:
#         return True

#     # 🔥 Pattern 3: plural intent
#     if re.search(r'\b(s|services|facilities|amenities)\b', q):
#         return True

#     return False




BINARY_INTENTS = {

    "breakfast_included": {

        "keywords": [
            "breakfast included",
            "complimentary breakfast",
            "free breakfast"
        ],

        "route": "binary"

    },

    "parking_available": {

        "keywords": [
            "parking",
            "car parking",
            "parking available"
        ],

        "route": "binary"

    },

    "wifi_available": {

        "keywords": [
            "wifi",
            "internet",
            "wireless internet"
        ],

        "route": "binary"

    }

}

def detect_binary_intent(query):

    return detect_intent(
        query,
        BINARY_INTENTS
    )

BINARY_ROUTE = {

    "rules":[

        {
            "type":"match",
            "pattern":r"^(is|are|do|does|can|will|should)\b"
        },

        {
            "type":"search",
            "pattern":r"(hai kya|milta hai|available hai)\??$"
        }

    ]

}


# def is_binary_question(q):

#     q = q.lower().strip()

#     # 🔥 Pattern 1: starts with auxiliary verb
#     if re.match(r'^(is|are|do|does|can|will|should)\b', q):
#         return True

#     # 🔥 Pattern 2: ends with question style (Hindi / Hinglish)
#     if re.search(r'(hai kya|milta hai|available hai)\??$', q):
#         return True

#     return False

def is_binary_query(query):

    return matches_route(
        query,
        BINARY_ROUTE
    )


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

CONTACT_ROUTE = {

    "rules":[

        {
            "type":"search",
            "pattern":r"\bcontact\b"
        },

        {
            "type":"search",
            "pattern":r"\bcontact number\b"
        },

        {
            "type":"search",
            "pattern":r"\bphone\b"
        },

        {
            "type":"search",
            "pattern":r"\bphone number\b"
        },

        {
            "type":"search",
            "pattern":r"\bmobile\b"
        },
        
        {
            "type":"search",
            "pattern":r"\bcall\b"
        },
        
        {
            "type":"search",
            "pattern":r"\breach\b"
        },

        {
            "type":"search",
            "pattern":r"\btelephone\b"
        }

    ]

}

# CONTACT_ROUTE = {

#     "patterns":[

#         "contact",
#         "contact number",
#         "phone",
#         "phone number",
#         "mobile",
#         "call",
#         "reach",
#         "telephone"

#     ]

# }

# def is_contact_query(query):

#     q = query.lower().strip()

#     return any(
#         pattern in q
#         for pattern in CONTACT_ROUTE["patterns"]
#     )

def is_contact_query(query):

    return matches_route(
        query,
        CONTACT_ROUTE
    )


CONTACT_INTENTS = {

    "phone_number":{

        "keywords":[
            "phone",
            "phone number",
            "call",
            "mobile"
        ],

        "route":"contact"

    },

    "whatsapp_number":{

        "keywords":[
            "whatsapp",
            "whatsapp number"
        ],

        "route":"contact"

    }

}

def detect_contact_intent(query):

    return detect_intent(
        query,
        CONTACT_INTENTS
    )

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


ACTION_ROUTE = {

    "rules":[

        {
            "type":"search",
            "pattern":r"\bneed\b"
        },

        {
            "type":"search",
            "pattern":r"\bwant\b"
        },

        {
            "type":"search",
            "pattern":r"\bsend\b"
        },

        {
            "type":"search",
            "pattern":r"\bbring\b"
        },

        {
            "type":"search",
            "pattern":r"\bprovide\b"
        },
        
        {
            "type":"search",
            "pattern":r"\bdeliver\b"
        },
        
        {
            "type":"search",
            "pattern":r"\barrange\b"
        },

        {
            "type":"search",
            "pattern":r"\bbook\b"
        },

        {
            "type":"search",
            "pattern":r"please\s+send"
        },

        {
            "type":"search",
            "pattern":r"please\s+bring"
        },

        {
            "type":"search",
            "pattern":r"i\s+need"
        },

        {
            "type":"search",
            "pattern":r"i\s+want"
        }

    ]

}


# ACTION_INTENTS = {

#     "request_towel": {
#         "keywords": [
#             "towel",
#             "towels"
#         ],
#         "route": "action"
#     },

#     "request_blanket": {
#         "keywords": [
#             "blanket",
#             "blankets"
#         ],
#         "route": "action"
#     },

#     "request_water": {
#         "keywords": [
#             "water",
#             "water bottle",
#             "water bottles"
#         ],
#         "route": "action"
#     },

#     "request_housekeeping": {
#         "keywords": [
#             "cleaning",
#             "housekeeping"
#         ],
#         "route": "action"
#     },

#     "request_tea": {
#         "keywords": [
#             "tea",
#             "cup tea"
#         ],
#         "route": "action"
#     },

#     "request_coffee": {
#         "keywords": [
#             "coffee"
#         ],
#         "route": "action"
#     },

#     "request_food": {
#         "keywords": [
#             "food",
#             "meal"
#         ],
#         "route": "action"
#     }

# }

def format_action_item(
    intent,
    quantity,
    unit=None
):

    config = ACTION_INTENTS[intent]

    item = config["keywords"][0]

    quantity_mode = config.get(
        "quantity_mode",
        "countable"
    )

    # --------------------------------
    # COUNTABLE
    # 2 towels
    # 1 towel
    # --------------------------------

    if quantity_mode == "countable":

        if quantity == 1:
            return f"{quantity} {item}"

        if not item.endswith("s"):
            item = f"{item}s"

        return f"{quantity} {item}"

    # --------------------------------
    # UNIT BASED
    # 2 cups of tea
    # 3 bottles of water
    # --------------------------------

    unit = (
        unit
        or config.get("default_unit")
    )

    if quantity == 1:

        unit = unit.rstrip("s")

    elif not unit.endswith("s"):

        unit = f"{unit}s"

    return (
        f"{quantity} {unit} of {item}"
    )


ACTION_INTENTS = {

    "request_towel": {
        "keywords": [
            "towel",
            "towels"
        ],
        "route": "action",

        "action": {
            "label": "Request Towel",
            "request_type": "towel",
            "default_quantity": 1,
            "message_template": "Please send {item_text} to my room."
        }
    },

    "request_water": {
        "keywords": [
            "water",
            "water bottle",
            "water bottles"
        ],
        "route": "action",

        "action": {
            "label": "Request Water",
            "request_type": "water",
            "default_quantity": 1,
            "message_template": "Please send {item_text} to my room."
        }
    },

    "request_tea": {
        "keywords": [
            "tea",
            "cup tea",
            "cups of tea"
        ],
        "route": "action",

        "action": {
            "label": "Request Tea",
            "request_type": "tea",
            "default_quantity": 1,
            "message_template": "Please send {item_text} to my room."
        }
    },

    "request_coffee": {
        "keywords": [
            "coffee",
            "cup coffee"
        ],
        "route": "action",

        "action": {
            "label": "Request Coffee",
            "request_type": "coffee",
            "default_quantity": 1,
            "message_template": "Please send {item_text} to my room."
        }
    },

    "request_food": {
        "keywords": [
            "food",
            "meal",
            "breakfast"
        ],
        "route": "action",

        "action": {
            "label": "Order Food",
            "request_type": "food",
            "default_quantity": 1,
            "message_template": "I would like to order {item_text}."
        }
    },

    "request_blanket": {
        "keywords": [
            "blanket",
            "blankets"
        ],
        "route": "action",

        "action": {
            "label": "Request Blanket",
            "request_type": "blanket",
            "default_quantity": 1,
            "message_template": "Please send {item_text} to my room."
        }
    }
}


def is_hotel_action_query(query):

    return matches_route(
            query,
            ACTION_ROUTE
        )



def detect_action_intent(query):

    q = query.lower()

    for intent, config in ACTION_INTENTS.items():

        for keyword in config["keywords"]:

            if keyword in q:
                return intent

    return None



def match_action_intent(query):

    q = query.lower().strip()

    for intent_name, config in ACTION_INTENTS.items():

        for keyword in config.get("keywords", []):

            if keyword in q:
                return intent_name

    return None


def is_implicit_action_request(query):

    q = query.lower().strip()

    matched_intent = match_action_intent(q)

    if not matched_intent:
        return None

    # Case 1: exact shorthand
    if any(
        q == keyword
        for keyword in ACTION_INTENTS[matched_intent]["keywords"]
    ):
        return matched_intent

    # Case 2: quantity-based request
    if re.search(r"\b\d+\b", q):
        return matched_intent

    return None


import re


def extract_action_quantity(query):

    q = query.lower().strip()

    patterns = [

        # 2 cups of tea
        r"\b(\d+)\s+(cups?|plates?|bottles?|glasses?|pieces?)\s+(?:of\s+)?",

        # 2 tea
        r"\b(\d+)\s+"
    ]

    for pattern in patterns:

        match = re.search(pattern, q)

        if match:

            quantity = int(match.group(1))

            if len(match.groups()) >= 2 and match.group(2):
                unit = match.group(2)
            else:
                unit = None

            return {
                "quantity": quantity,
                "unit": unit
            }

    return {
        "quantity": None,
        "unit": None
    }


def extract_action_details(query, intent):

    quantity_info = extract_action_quantity(query)

    quantity = quantity_info["quantity"]
    unit = quantity_info["unit"]

    if not quantity:
        quantity = 1

    return {
        "intent": intent,
        "quantity": quantity,
        "unit": unit
    }



INTENT_REGISTRIES = {

    "time": TIME_INTENTS,

    "binary": BINARY_INTENTS,

    "contact": CONTACT_INTENTS,

    "list": LIST_INTENTS,

    "action": ACTION_INTENTS

}

def detect_intent(query, intent_registry):

    print("=" * 50)
    print("🧩 🧩 🧩 detect_intent CALLED🧩 🧩 🧩 ")
    print("QUERY:", query)
    print("ROUTE TYPE:", type(intent_registry))
    print("ROUTE VALUE:", intent_registry)
    print("=" * 50)

    q = query.lower().strip()
    # registry = INTENT_REGISTRIES.get(route, {})

    for intent_name, config in intent_registry.items():

        keywords = config.get("keywords", [])

        for keyword in keywords:
            if keyword in q:
                return intent_name
    return None

def get_intent_registry(route):
    return INTENT_REGISTRIES.get(route, {})

def detect_route_intent(query, route):

    registry = get_intent_registry(route)

    return detect_intent(query, registry)




DOMAIN_ENTITY_REGISTRY = {

    "wifi": {
        "keywords": [
            "wifi",
            "wi-fi",
            "internet",
            "wireless",
            "network"
        ],
        "businesses": ["hotel", "restaurant", "clinic"],
        "section": "services",
        "display_name": "wi fi"
    },

    "laundry": {
        "keywords": [
            "laundry",
            "dry cleaning",
            "washing",
            "clothes wash"
        ],
        "businesses": ["hotel"],
        "section": "services",
        "display_name": "laundry"
    },

    "swimming pool": {
        "keywords": [
            "swimming pool",
            "pool",
            "swimming",
            "indoor pool",
            "outdoor pool"
        ],
        "businesses": ["hotel", "resort"],
        "section": "amenities",
        "display_name": "swimming pool"
    },

    "gym": {
        "keywords": [
            "gym",
            "fitness center",
            "fitness",
            "workout",
            "exercise room"
        ],
        "businesses": ["hotel"],
        "section": "amenities",
        "display_name": "gym and fitness"
    },

    "spa": {
        "keywords": [
            "spa",
            "massage",
            "wellness",
            "spa center"
        ],
        "businesses": ["hotel", "resort"],
        "section": "amenities",
        "display_name": "spa"
    },

    "restaurant": {
        "keywords": [
            "restaurant",
            "dining",
            "restaurant service",
            "eat"
        ],
        "businesses": ["hotel", "restaurant"],
        "section": "dining",
        "display_name": "restaurant"
    },

    "breakfast": {
        "keywords": [
            "breakfast",
            "morning buffet",
            "buffet",
            "morning meal"
        ],
        "businesses": ["hotel"],
        "section": "dining",
        "display_name": "breakfast"
    },

    "parking": {
        "keywords": [
            "parking",
            "car parking",
            "parking area",
            "vehicle parking"
        ],
        "businesses": ["hotel", "restaurant", "clinic"],
        "section": "facilities",
        "display_name": "parking"
    },

    "room_service": {
        "keywords": [
            "room service",
            "food service",
            "in-room dining",
            "room delivery"
        ],
        "businesses": ["hotel"],
        "section": "services",
        "display_name": "room service"
    },

    "room": {
        "keywords": [
            "rooms",
            "delux rooms",
            "in-room dining",
            "room delivery"
        ],
        "businesses": ["hotel"],
        "section": "facilities",
        "display_name": "different types of rooms"
    },

    "housekeeping": {
        "keywords": [
            "housekeeping",
            "cleaning",
            "room cleaning",
            "clean room"
        ],
        "businesses": ["hotel"],
        "section": "services",
        "display_name": "house-keeping"
    },

    "elevator": {
        "keywords": [
            "elevator",
            "lift"
        ],
        "businesses": ["hotel", "clinic"],
        "section": "facilities",
        "display_name": "elevator"
    },

    "airport_pickup": {
        "keywords": [
            "airport pickup",
            "airport drop",
            "airport transfer",
            "pickup",
            "drop"
        ],
        "businesses": ["hotel"],
        "section": "transport",
        "display_name": "airport pick-up"
    },

    "conference_room": {
        "keywords": [
            "conference room",
            "meeting room",
            "conference hall",
            "meeting hall",
            "board room"
        ],
        "businesses": ["hotel"],
        "section": "business",
        "display_name": "conference room"
    },

    "banquet_hall": {
        "keywords": [
            "banquet hall",
            "banquet",
            "event hall",
            "wedding hall",
            "party hall"
        ],
        "businesses": ["hotel"],
        "section": "events",
        "display_name": "banquet hall"
    },

    "coffee_shop": {
        "keywords": [
            "coffee shop",
            "cafe",
            "coffee",
            "bakery"
        ],
        "businesses": ["hotel", "restaurant"],
        "section": "dining",
        "display_name": "coffee shop"
    }

}



def resolve_entity_from_text(text, registry):
    text = text.lower().strip()

    matches = []

    for entity, config in registry.items():

        keywords = config.get("keywords", [])

        for keyword in keywords:

            keyword = keyword.lower().strip()

            if keyword and keyword in text:
                matches.append(
                    (len(keyword), entity)
                )

    if not matches:
        return None

    # Prefer the longest matching phrase.
    # Example:
    # "swimming pool" should beat "pool"
    matches.sort(reverse=True)

    return matches[0][1]








def detect_entities(query, registry):

    q = query.lower().strip()

    entities = []

    for entity_name, config in registry.items():

        keywords = config.get("keywords", [])

        for keyword in keywords:

            if keyword in q:

                entities.append(entity_name)

                break

    return entities


def detect_domain_entities(query):

    return detect_entities(
        query,
        DOMAIN_ENTITY_REGISTRY
    )

SCORING_CONFIG = {

        "intent_bonus": 2.0,

        "section_bonus": 0.3,

        "list_bonus": 1.0,

        "pattern_bonus": 1.0,

        "vector_weight": 0.6,

        "bm25_weight": 0.4,

        "entity_bonus": 2.0

    }



def calculate_entity_bonus(
    text,
    entities,
    registry
):
    # print("CURRENT ENTITY:", entities)
    text = text.lower()

    entity_bonus = 0.0

    for entity in entities:

        config = registry.get(entity)
        
        # print("CONFIG:", config)

        if not config:
            continue

        keywords = config.get("keywords", [])

        for keyword in keywords:

            # print("KEYWORDS:", keywords)

            if keyword in text:
                if keyword in text:
                    print("MATCH FOUND:", keyword)

                entity_bonus += SCORING_CONFIG["entity_bonus"]

                break

    return entity_bonus


def calculate_domain_entity_bonus(
    text,
    entities
):

    # print("=" * 50)
    # print("TEXT:", text)
    # print("ENTITIES:", entities)

    return calculate_entity_bonus(
        text,
        entities,
        DOMAIN_ENTITY_REGISTRY
    )



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

    if is_binary_query(q) or is_binary_query(oq):
        return "binary"
    
    if is_hotel_action_query(oq) or is_implicit_action_request(oq):
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



def entity_has_evidence(entity, chunks):

    config = DOMAIN_ENTITY_REGISTRY.get(entity)

    if not config:
        return False

    keywords = config.get("keywords", [])

    for chunk in chunks:

        text = chunk.get("text", "").lower()

        for keyword in keywords:

            if keyword.lower() in text:
                return True

    return False