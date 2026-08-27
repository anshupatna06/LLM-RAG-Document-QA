# import re
# from agent.utils.query_classifier import matches_route
# # HOTEL_SERVICES = [

# #     "towel",
# #     "blanket",
# #     "pillow",
# #     "water",
# #     "water bottle",

# #     "wifi",
# #     "internet",

# #     "food",
# #     "breakfast",
# #     "lunch",
# #     "dinner",

# #     "cleaning",
# #     "housekeeping",

# #     "ac",
# #     "tv"
# # ]



# # REQUEST_WORDS = [

# #     "need",
# #     "send",
# #     "bring",
# #     "help",
# #     "request",
# #     "provide",
# #     "want",

# #     "deliver",
# #     "arrange",
# #     "book",
# #     "call",

# #     "please",
# #     "urgent"
# # ]


# # ACTION_PATTERNS = [

# #     r"\bneed\b",
# #     r"\bwant\b",

# #     r"\bsend\b",
# #     r"\bbring\b",

# #     r"\bprovide\b",
# #     r"\bdeliver\b",

# #     r"\barrange\b",
# #     r"\bbook\b",

# #     r"please\s+send",
# #     r"please\s+bring",

# #     r"can\s+you\s+send",
# #     r"can\s+you\s+bring",

# #     r"i\s+need",
# #     r"i\s+want"
# # ]


# # ACTION_ONLY_SERVICES = [

# #     "towel",
# #     "blanket",
# #     "pillow",

# #     "water",
# #     "water bottle"
# # ]



# # def is_hotel_action_query(query):

# #     q = query.lower().strip()

# #     if q in ACTION_ONLY_SERVICES:
# #         return True

# #     has_request = any(w in q for w in REQUEST_WORDS)

# #     has_service = any(s in q for s in HOTEL_SERVICES)

    

# #     #is_info = any(p in q for p in informational_patterns)
# #     has_action_pattern = any(re.search(pattern, q) for pattern in ACTION_PATTERNS)

# #     if has_request and has_service:
# #         return True

# #     # if has_service and len(q.split()) <= 3 and not is_info:
# #     #     return True
# #     if has_action_pattern and has_service:
# #         return True

# #     return False



# # ACTION_ROUTE = {

# #     "patterns":[

# #         r"\bneed\b",
# #         r"\bwant\b",

# #         r"\bsend\b",
# #         r"\bbring\b",

# #         r"\bprovide\b",
# #         r"\bdeliver\b",

# #         r"\barrange\b",
# #         r"\bbook\b",

# #         r"please\s+send",
# #         r"please\s+bring",

# #         r"can\s+you\s+send",
# #         r"can\s+you\s+bring",

# #         r"i\s+need",
# #         r"i\s+want"

# #     ]

# # }

# # ACTION_INTENTS = {

# #     "request_towel":{

# #         "keywords":[
# #             "towel"
# #         ],

# #         "route":"action"

# #     },

# #     "request_blanket":{

# #         "keywords":[
# #             "blanket"
# #         ],

# #         "route":"action"

# #     },

# #     "request_water":{

# #         "keywords":[
# #             "water",
# #             "water bottle"
# #         ],

# #         "route":"action"

# #     },

# #     "request_housekeeping":{

# #         "keywords":[
# #             "cleaning",
# #             "housekeeping"
# #         ],

# #         "route":"action"

# #     }

# # }


# # def is_hotel_action_query(query):

# #     q = query.lower().strip()

# #     return any(
# #         re.search(pattern, q)
# #         for pattern in ACTION_ROUTE["patterns"]
# #     )


