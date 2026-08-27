from datetime import datetime

SERVICE_REQUESTS = []
REQUEST_COUNTER = 1000

ALLOWED_STATUS = {
    "pending",
    "in_progress",
    "completed"
}


def create_service_request(room, request_type, client_id, quantity, unit, display_text):

    global REQUEST_COUNTER
    REQUEST_COUNTER += 1

    request = {

        "request_id": f"REQ-{REQUEST_COUNTER}", 
        "room": room,
        "request": request_type,
        "client_id": client_id,
        "quantity": quantity,
        "unit": unit,
        "display_text":display_text,
        "status": "pending",
        "time": datetime.now().strftime("%I:%M %p")
    }

    SERVICE_REQUESTS.append(request)
    print("🔥 NEW SERVICE REQUEST:", request)

    return request


def get_all_requests():

    return SERVICE_REQUESTS


VALID_TRANSITIONS = {
    "pending": ["in_progress"],
    "in_progress": ["completed"],
    "completed": []
}

# def update_request_status(request_id, new_status):

#     for req in SERVICE_REQUESTS:

#         if req["request_id"] == request_id:

#             req["status"] = new_status

#             return req

#     return None

def update_request_status(request_id, new_status):

    for req in SERVICE_REQUESTS:

        if req["request_id"] == request_id:

            current_status = req["status"]

            allowed_next_statuses = VALID_TRANSITIONS.get(
                current_status,
                []
            )

            if new_status not in allowed_next_statuses:

                return {
                    "success": False,
                    "reason": "invalid_transition",
                    "current_status": current_status
                }

            req["status"] = new_status

            return {
                "success": True,
                "request": req
            }

    return {
        "success": False,
        "reason": "request_not_found"
    }