def detect_intent(query):

    q = query.lower()

    if any(w in q for w in ["time", "when", "hours"]):
        return "time"

    if any(w in q for w in ["price", "cost", "charge"]):
        return "pricing"

    if any(w in q for w in ["available", "availability", "rooms"]):
        return "availability"

    if any(w in q for w in ["facility", "amenities", "services"]):
        return "facility"

    return "general"
