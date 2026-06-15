def normalize_speech_query(query):

    q = query.lower().strip()

    replacements = {

        # contact
        "contact no": "contact number",
        "contact now": "contact number",
        "phone no": "phone number",

        # wifi
        "why fi": "wifi",
        "wi fi": "wifi",

        # checkin
        "check in": "checkin",
        "check out": "checkout",

        # room service
        "room services": "room service",

        # breakfast
        "break fast": "breakfast",

        # water
        "water battle": "water bottle",

        # AC
        "a c": "ac"
    }

    for wrong, correct in replacements.items():
        q = q.replace(wrong, correct)

    return q