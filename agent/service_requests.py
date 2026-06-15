from datetime import datetime

SERVICE_REQUESTS = []
REQUEST_COUNTER = 1000


def create_service_request(room, request_type, client_id):

    global REQUEST_COUNTER
    REQUEST_COUNTER += 1

    request = {

        "request_id": f"REQ-{REQUEST_COUNTER}", 
        "room": room,
        "request": request_type,
        "client_id": client_id,
        "status": "pending",
        "time": datetime.now().strftime("%I:%M %p")
    }

    SERVICE_REQUESTS.append(request)
    print("🔥 NEW SERVICE REQUEST:", request)

    return request


def get_all_requests():

    return SERVICE_REQUESTS

def update_request_status(request_id, new_status):

    for req in SERVICE_REQUESTS:

        if req["request_id"] == request_id:

            req["status"] = new_status

            return req

    return None