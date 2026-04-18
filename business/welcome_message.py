def generate_welcome_message(config):

    name = config["name"]

    return {
    "message": f"""
👋 Welcome to {config['name']} Assistant

I can help you with hotel facilities, dining,
check-in details and services.

How can I assist you today?
""",
    "suggestions": [
        "Is breakfast included?",
        "What time is check-in?",
        "Do you have parking?"
    ]
}
