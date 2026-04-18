def detect_entity(query):

    q = query.lower()

    entity_map = {
        "breakfast": ["breakfast"],
        "parking": ["parking"],
        "wifi": ["wifi", "internet"],
        "checkin": ["check-in", "check in"],
        "room": ["room"],
        "consultation": ["consultation"],
        "diagnostic": ["diagnostic", "lab", "test"],
        "health checkups": ["Preventive health checkups", "health checkups"],
        "menu": ["menu", "food"],
        "timing": ["time", "hours", "open"]
    }

    for key, keywords in entity_map.items():
        if any(k in q for k in keywords):
            return key

    return None