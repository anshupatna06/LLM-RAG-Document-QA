import random
import re
from agent.utils.intent import detect_intent
from agent.utils.entity import detect_entity

SUGGESTION_CONFIG = {

    "hotel": {

        "breakfast": [
            "What time is breakfast served?",
            "Is breakfast complimentary?",
            "Where is breakfast served?",
            "Do you offer vegetarian breakfast?",
            "Is breakfast included in all rooms?"
        ],

        "parking": [
            "Is parking free?",
            "Is valet parking available?",
            "Is parking available overnight?"
        ],

        "wifi": [
            "Is WiFi free?",
            "Is WiFi available in rooms?",
            "Do you offer high-speed internet?"
        ],

        "checkin": [
            "What is check-in time?",
            "Can I request early check-in?",
            "Is late check-in available?"
        ],

        "room": [
            "What types of rooms are available?",
            "Do you have deluxe rooms?",
            "What is the price per night?"
        ]
    },

    "clinic": {

        "consultation": [
            "What are consultation hours?",
            "Can I book an appointment?",
            "Is walk-in consultation available?"
        ],

        "diagnostic": [
            "Do you offer diagnostic services?",
            "Are lab tests available?",
            "Do you provide health checkups?"
        ]
    },

    "restaurant": {

        "menu": [
            "What is on the menu?",
            "Do you have vegetarian options?",
            "Do you offer desserts?"
        ],

        "timing": [
            "What are opening hours?",
            "When do you close?",
            "Do you serve late night food?"
        ]
    }
}





def make_question(text):

    t = text.lower()

    if "wifi" in t:
        return "Is WiFi available?"

    if "parking" in t:
        return "Do you offer parking?"

    if "breakfast" in t:
        return "Is breakfast included?"

    if "consultation" in t:
        return "Do you offer consultations?"

    if "diagnostic" in t:
        return "Do you provide diagnostic services?"

    return f"Do you offer {t}?"


def generate_suggestions(
    retrieval=None,
    query=None,
    business_id=None,
    first_turn=False,
    max_suggestions=3
):

    BUSINESS_DEFAULTS = {

        "hotel": [
            "Is breakfast included?",
            "What facilities are available?",
            "Do you have parking?"
        ],

        "clinic": [
            "What services are available?",
            "Do you offer diagnostics?",
            "What are consultation hours?"
        ],

        "restaurant": [
            "What is on the menu?",
            "Do you have vegetarian options?",
            "What are opening hours?"
        ]
    }

    # -------------------
    # INTENT-BASED SUGGESTIONS (NEW 🔥)
    # -------------------
    # -------------------
    # INTENT + ENTITY + BUSINESS AWARE SUGGESTIONS 🔥
    # -------------------

    if query and business_id:

        # -------------------
        # 1. ENTITY-BASED (HIGHEST PRIORITY)
        # -------------------
        entity = detect_entity(query)

        if entity:
            business_data = SUGGESTION_CONFIG.get(business_id, {})

            if entity in business_data:
                suggestions = business_data[entity][:]

                random.shuffle(suggestions)
                return suggestions[:max_suggestions]


        # -------------------
        # 2. INTENT-BASED (FALLBACK)
        # -------------------
        intent = detect_intent(query)

        if business_id == "hotel":

            if intent == "time":
                suggestions = [
                    "What time is breakfast served?",
                    "What are check-in and check-out timings?",
                    "When does the restaurant open?"
                ]

            elif intent == "facility":
                suggestions = [
                    "What facilities are available?",
                    "Do you offer parking?",
                    "Is WiFi available?"
                ]

            elif intent == "availability":
                suggestions = [
                    "Are rooms available today?",
                    "Do you have deluxe rooms?",
                    "Is early booking required?"
                ]

            else:
                suggestions = []


        elif business_id == "clinic":
 
            if intent == "time":
                suggestions = [
                    "What are consultation hours?",
                    "When is the clinic open?",
                    "Are doctors available today?"
                ]

            elif intent == "facility":
                suggestions = [
                    "What services are available?",
                    "Do you offer diagnostics?",
                    "Do you provide emergency care?"
                ]

            elif intent == "availability":
                suggestions = [
                    "Is doctor available today?",
                    "Can I book an appointment?",
                    "Is walk-in consultation allowed?"
                ]

            else:
                suggestions = []


        elif business_id == "restaurant":
      
            if intent == "time":
                suggestions = [
                    "What are opening hours?",
                    "When do you close?",
                    "Do you serve late night food?"
                ]

            elif intent == "facility":
                suggestions = [
                    "What is on the menu?",
                    "Do you have vegetarian options?",
                    "Do you offer home delivery?"
                ]

            else:
                suggestions = []

        else:
            suggestions = []

        # shuffle intent suggestions too (important)
        if suggestions:
            random.shuffle(suggestions)
            return suggestions[:max_suggestions]
        
        
    # -----------------------------
    # FIRST TURN (NO QUERY)
    # -----------------------------
    if first_turn:
        return BUSINESS_DEFAULTS.get(business_id, [])[:max_suggestions]

    # -------------------
    # RETRIEVAL-BASED SUGGESTIONS (SMART)
    # -------------------
    if retrieval:

        suggestions = []

        for chunk in retrieval.get("chunks", []):

            if not chunk.get("used"):
                continue

            facility_split = re.split(r'(?:\n|•|-|\.)', chunk["text"])

            for item in facility_split:

                clean = item.strip()

                if len(clean.split()) < 3:
                    continue

                # BUSINESS FILTERING
                if business_id == "clinic":
                    if not any(w in clean.lower() for w in [
                        "service", "consult", "diagnostic",
                        "treatment", "lab", "vaccination", "checkup"
                    ]):
                        continue

                if business_id == "hotel":
                    if not any(w in clean.lower() for w in [
                        "room", "wifi", "parking", "breakfast"
                    ]):
                        continue

                if 15 < len(clean) < 80:
                    suggestions.append(make_question(clean))

        # remove duplicates + limit
        suggestions = list(dict.fromkeys(suggestions))

        if suggestions:
            return suggestions[:max_suggestions]

    # -------------------
    # FALLBACK
    # -------------------
    return BUSINESS_DEFAULTS.get(business_id, [])[:max_suggestions]