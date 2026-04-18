def detect_business(question):

    q = question.lower()

    clinic_keywords = [
        "doctor","clinic","cardiology","appointment",
        "patient","consultation","medical"
    ]

    restaurant_keywords = [
        "menu","dish","food","restaurant","meal",
        "reservation","dinner"
    ]

    hotel_keywords = [
        "room","check-in","check out","stay",
        "hotel","breakfast","facilities"
    ]

    if any(k in q for k in clinic_keywords):
        return "clinic"

    if any(k in q for k in restaurant_keywords):
        return "restaurant"

    return "hotel"