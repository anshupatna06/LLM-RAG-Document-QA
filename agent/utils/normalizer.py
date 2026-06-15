HINGLISH_MAP = {
    "subidha": "suvidha",
    "suvidha": "facilities",
    "facility": "facilities",
    "wifi": "wifi",
    "net": "internet",
    "khana": "food",
    "nashta": "breakfast"
}

def normalize_local_query(query):

    q = query.lower()

    for k, v in HINGLISH_MAP.items():
        if k in q:
            q = q.replace(k, v)

    return q