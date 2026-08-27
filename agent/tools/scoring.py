# import re
# from agent.utils.query_classifier import INTENT_EVIDENCE_REGISTRY, calculate_domain_entity_bonus, is_list_query, is_time_query


# query_context = {
#     "query": query,
#     "route": query_type,
#     "intent": intent,
#     "entities": entities,
#     "target_section": target_section,
#     "is_list": is_list_query(query),
#     "is_time": is_time_query(query),
# }

# SCORING_CONFIG = {

#         "intent_bonus": 2.0,

#         "section_bonus": 0.3,

#         "list_bonus": 1.0,

#         "pattern_bonus": 1.0,

#         "vector_weight": 0.6,

#         "bm25_weight": 0.4,

#         "entity_bonus": 2.0

#     }

# def calculate_retrieval_score(chunk, query_context):
#     return (
#         chunk["vector_score"] * SCORING_CONFIG["vector_weight"]
#         +
#         chunk["bm25_score"] * SCORING_CONFIG["bm25_weight"]
#     )

# def calculate_intent_evidence(chunk, query_context):

#     intent = query_context.get("intent")

#     if not intent:
#         return 0.0

#     config = INTENT_EVIDENCE_REGISTRY.get(intent)

#     if not config:
#         return 0.0

#     text = chunk["text"].lower()

#     for keyword in config.get("keywords", []):

#         if keyword in text:
#             return config.get("bonus", 0.0)

#     return 0.0


# def calculate_entity_evidence(chunk, query_context):

#     entities = query_context.get("entities", [])

#     if not entities:
#         return 0.0

#     return calculate_domain_entity_bonus(
#         chunk["text"],
#         entities
#     )


# # def calculate_section_evidence(chunk, query_context):

# #     target_section = query_context.get("target_section")

# #     if not target_section:
# #         return 0.0

# #     if chunk["section"] == target_section:
# #         return SCORING_CONFIG["section_bonus"]

# #     return 0.0
# def calculate_section_evidence(chunk, query_context):

#     target_section = query_context.get("target_section")

#     if not target_section:
#         return 0.0

#     section = chunk.get("section", "")

#     if target_section in section:
#         return SCORING_CONFIG["section_bonus"]

#     return 0.0
# # def calculate_list_evidence(chunk, query_context):

# #     if not query_context.get("is_list"):
# #         return 0.0

# #     if chunk.get("list_title"):
# #         return SCORING_CONFIG["list_bonus"]

# #     return 0.0
# def calculate_list_evidence(chunk, query_context):

#     if not query_context.get("is_list"):
#         return 0.0

#     if re.search(
#         r"\d{1,2}:\d{2}",
#         chunk["text"]
#     ):
#         return SCORING_CONFIG["list_bonus"]

#     return 0.0


# def calculate_pattern_evidence(chunk, query_context):

#     query = query_context["query"]

#     if is_time_query(query):

#         if re.search(
#             r"\d{1,2}:\d{2}|closed",
#             chunk["text"],
#             re.I
#         ):
#             return SCORING_CONFIG["pattern_bonus"]

#     return 0.0

# FEATURE_SCORERS = [
#     calculate_retrieval_score,
#     calculate_intent_evidence,
#     calculate_entity_evidence,
#     calculate_section_evidence,
#     calculate_list_evidence,
#     calculate_pattern_evidence,
# ]

# def calculate_final_score(chunk, query_context):

#     final_score = 0.0

#     for scorer in FEATURE_SCORERS:

#         final_score += scorer(
#             chunk,
#             query_context
#         )

#     return final_score

import re

from agent.utils.query_classifier import (
    INTENT_EVIDENCE_REGISTRY,
    calculate_domain_entity_bonus,
)


SCORING_CONFIG = {

    "intent_bonus": 2.0,
    "section_bonus": 0.3,
    "list_bonus": 1.0,
    "pattern_bonus": 1.0,

    "vector_weight": 0.6,
    "bm25_weight": 0.4,

    "entity_bonus": 2.0
}


def calculate_retrieval_score(chunk, query_context):

    return (
        chunk["vector_score"]
        * SCORING_CONFIG["vector_weight"]

        +

        chunk["bm25_score"]
        * SCORING_CONFIG["bm25_weight"]
    )


def calculate_intent_evidence(chunk, query_context):

    intent = query_context.get("intent")

    if not intent:
        return 0.0

    config = INTENT_EVIDENCE_REGISTRY.get(intent)

    if not config:
        return 0.0

    text = chunk["text"].lower()

    for keyword in config.get("keywords", []):

        if keyword in text:
            return config.get("bonus", 0.0)

    return 0.0


def calculate_entity_evidence(chunk, query_context):

    entities = query_context.get("entities", [])

    if not entities:
        return 0.0

    return calculate_domain_entity_bonus(
        chunk["text"],
        entities
    )


def calculate_section_evidence(chunk, query_context):

    target_section = query_context.get("target_section")
    # print("🚥🚥🚥🚥TARGET SECTION IN EVIDNCE FUNCTION FROM QUERY SIDE:", target_section)

    if not target_section:
        return 0.0

    section = chunk.get("section", "")
    # print("TARGET SECTION DETECTED IN CHUNK SIDE:", section)

    if target_section in section:
        # print("TARGET SECTION DETECD IN SECTION:",target_section)
        return SCORING_CONFIG["section_bonus"]

    return 0.0


def calculate_list_evidence(chunk, query_context):

    if not query_context.get("is_list"):
        return 0.0

    # Preserve OLD algorithm for now.
    if re.search(
        r"\d{1,2}:\d{2}",
        chunk["text"]
    ):
        return SCORING_CONFIG["list_bonus"]

    return 0.0


def calculate_pattern_evidence(chunk, query_context):

    # Old implementation was commented out.
    # Keep behavior unchanged during refactoring.

    return 0.0


def calculate_lexical_evidence(
    chunk,
    query_context
):

    query = query_context.get("query", "").lower()

    text = chunk.get("text", "").lower()

    query_words = set(query.split())
    chunk_words = set(text.split())

    overlap = len(query_words & chunk_words)

    return overlap * 0.1


FEATURE_SCORERS = [

    calculate_retrieval_score,
    calculate_intent_evidence,
    calculate_entity_evidence,
    calculate_section_evidence,
    calculate_list_evidence,
    calculate_pattern_evidence,
    calculate_lexical_evidence,
]


# def calculate_final_score(chunk, query_context):

#     final_score = 0.0

#     for scorer in FEATURE_SCORERS:

#         final_score += scorer(
#             chunk,
#             query_context
#         )
#         print("="*70)
#         print("FINAL:", final_score)
#         print("="*70)

#     return final_score
    

def calculate_final_score(chunk, query_context):

    final_score = 0.0

    print("\n" + "=" * 70)
    print("SCORING CHUNK:")
    print(chunk["text"][:100])
    print("=" * 70)

    for scorer in FEATURE_SCORERS:

        score = scorer(
            chunk,
            query_context
        )

        final_score += score

        print(
            f"{scorer.__name__:<35} "
            f"{score:+.2f}"
        )

    print("-" * 70)
    print(f"FINAL SCORE: {final_score:.2f}")
    print("=" * 70)

    return final_score



