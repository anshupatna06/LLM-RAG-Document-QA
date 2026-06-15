from agent.utils.message_formatter import format_whatsapp_message
def generate_actions(answer, query_type, query):

    q = query.lower()
    actions = []

    # -------------------------
    # WIFI
    # -------------------------
    if "wifi" in q:
        actions.append({
            "label": "Help me connect",
            "type": "assist_wifi"
        })

        actions.append({
            "label": "Contact Reception",
            "type": "call"
        })

    # -------------------------
    # TOWEL / SERVICE
    # -------------------------
    elif "towel" in q or "room service" in q:
        actions.append({
            "label": "Request Towel",
            "type": "whatsapp",
            "message": "Please send a towel to my room."
        })

    # -------------------------
    # FOOD
    # -------------------------
    elif "food" in q or "breakfast" in q:
        actions.append({
            "label": "Order Food",
            "type": "whatsapp",
            "message": "I want to order food."
        })

    # -------------------------
    # DEFAULT
    # -------------------------

    if not actions:
        actions.append({
            "label": "Ask on WhatsApp",
            "type": "whatsapp",
            "message": format_whatsapp_message(query, query_type)
        })
    
        actions.append({
            "label": "Call Reception",
            "type": "call"
        })

    return actions



