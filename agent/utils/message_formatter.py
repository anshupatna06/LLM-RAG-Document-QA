def format_whatsapp_message(query, route="general"):

    q = query.lower()

    # -------------------------
    # TIME QUERIES
    # -------------------------
    if route == "time":
        return f"Hi, I am staying at your hotel. I want to confirm {query.lower()}."

    # -------------------------
    # WIFI
    # -------------------------
    if "wifi" in q:
        return "Hi, I am unable to connect to the WiFi. Please help me."

    # -------------------------
    # TOWEL
    # -------------------------
    if "towel" in q:
        return "Hi, I need fresh towels in my room."

    # -------------------------
    # FOOD
    # -------------------------
    if "food" in q or "room service" in q:
        return "Hi, I would like assistance with food or room service."

    # -------------------------
    # PARKING
    # -------------------------
    if "parking" in q:
        return "Hi, I want to ask about parking availability."

    # -------------------------
    # FALLBACK
    # -------------------------
    return f"Hi, I need assistance regarding: {query}"