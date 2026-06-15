from agent.utils.message_formatter import format_whatsapp_message


def generate_hotel_actions(query, route):

    q = query.lower()

    actions = []

    # towel
    if "towel" in q:

        actions.append({
            "label": "Request Towel",
            "type": "whatsapp",
            "request_type": "towel",
            "message": "Please send towels to my room."
        })

    # water
    elif "water" in q:

        actions.append({
            "label": "Request Water",
            "type": "whatsapp",
            "request_type": "water",
            "message": "Please send water bottles to my room."
        })

    # wifi
    elif "wifi" in q:

        actions.append({
            "label": "Help me connect",
            "type": "assist_wifi"
        })

    # food
    elif "food" in q or "breakfast" in q:

        actions.append({
            "label": "Order Food",
            "type": "whatsapp",
            "request_type": "food",
            "message": "I want to order food."
        })

    # default
    if not actions:

        actions.append({
            "label": "Ask on WhatsApp",
            "type": "whatsapp",
            "message": format_whatsapp_message(query, route)
        })

        actions.append({
            "label": "Call Reception",
            "type": "call"
        })

    return actions

def extract_service_type(query):

    q = query.lower()

    if "towel" in q:
        return "towel"

    if "water" in q:
        return "water"

    if "wifi" in q:
        return "wifi"

    if "food" in q:
        return "food"

    return "general"
