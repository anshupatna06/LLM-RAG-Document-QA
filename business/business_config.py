from business.hotel_config import HOTEL_CONFIG
from business.restaurant_config import RESTAURANT_CONFIG
from business.clinic_config import CLINIC_CONFIG


BUSINESSES = {
    "hotel": HOTEL_CONFIG,
    "restaurant": RESTAURANT_CONFIG,
    "clinic": CLINIC_CONFIG
}


def get_business_config(business_id):

    return BUSINESSES.get(business_id, HOTEL_CONFIG)