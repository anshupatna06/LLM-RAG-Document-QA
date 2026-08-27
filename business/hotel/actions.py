from agent.utils.message_formatter import format_whatsapp_message
from agent.utils.query_classifier import extract_action_quantity, ACTION_INTENTS, detect_action_intent, format_action_item

# def generate_hotel_actions(query, route):

#     q = query.lower()

#     actions = []

#     # towel
#     if "towel" in q:

#         actions.append({
#             "label": "Request Towel",
#             "type": "whatsapp",
#             "request_type": "towel",
#             "message": "Please send towels to my room."
#         })

#     # water
#     elif "water" in q:

#         actions.append({
#             "label": "Request Water",
#             "type": "whatsapp",
#             "request_type": "water",
#             "message": "Please send water bottles to my room."
#         })

#     elif "tea" in q:
    
#         actions.append({
#             "label": "Request tea",
#             "type": "whatsapp",
#             "request_type": "tea",
#             "message": "Please send tea to my room."
#         })

#     # wifi
#     elif "wifi" in q:

#         actions.append({
#             "label": "Help me connect",
#             "type": "assist_wifi"
#         })

#     # food
#     elif "food" in q or "breakfast" in q:

#         actions.append({
#             "label": "Order Food",
#             "type": "whatsapp",
#             "request_type": "food",
#             "message": "I want to order food."
#         })

#     # default
#     if not actions:

#         actions.append({
#             "label": "Ask on WhatsApp",
#             "type": "whatsapp",
#             "message": format_whatsapp_message(query, route)
#         })

#         actions.append({
#             "label": "Call Reception",
#             "type": "call"
#         })

#     return actions



def generate_hotel_actions(
    query,
    route,
    intent=None
):

    actions = []

    # ------------------------------------------
    # ACTION INTENT FOUND
    # ------------------------------------------
    intent = detect_action_intent(query)
    

    if route == "action" and intent:
        

        config = ACTION_INTENTS.get(intent)

        if config:

            action_config = config.get("action", {})

            quantity_info = extract_action_quantity(query)

            quantity = (
                quantity_info["quantity"]
                or action_config.get(
                    "default_quantity",
                    1
                )
            )

            unit = quantity_info["unit"]

            item = config["keywords"][0]
            print("ITEM DETECTED IN ACTION: ", item)

            # # Preserve quantity + unit
            # if unit:

            #     # item_text = (
            #     #     f"{quantity} {unit} of {item}"
            #     # )
            #     item_text = format_action_item(intent, quantity, unit)
            #     print("ITEM TEXT INSIDE UNIT: ", item_text)

            # else:

            #     # Handle singular/plural later
            #     item_text = (
            #         f"{quantity} {item}"
            #     )
            item_text = format_action_item(intent, quantity, unit)
            print("FORMATTED ITEM TEXT:", item_text)

            message = (
                action_config
                .get(
                    "message_template",
                    "Please send {item_text} to my room."
                )
                .format(
                    item_text=item_text
                )
            )

            actions.append({

                "label": action_config.get(
                    "label",
                    "Send Request"
                ),

                "type": "whatsapp",

                "request_type": action_config.get(
                    "request_type",
                    intent
                ),

                # 🔥 IMPORTANT:
                # Preserve structured information
                "quantity": quantity,
                "unit": unit,
                "display_text": item_text,

                "message": message
            })

            return actions

    # ------------------------------------------
    # FALLBACK
    # ------------------------------------------
    
    actions.append({

        "label": "Ask on WhatsApp",

        "type": "whatsapp",

        "message": format_whatsapp_message(
            query,
            route
        )
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
