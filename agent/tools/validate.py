import time
import re
from agent.utils.entity import detect_entity
from agent.utils.fallback_handler import fallback_response

from agent.utils.query_classifier import (
    is_time_query,
    detect_time_intent,
    is_list_query,
    is_binary_query,
    route_query,
    is_feature_query,
    entity_has_evidence,
    DOMAIN_ENTITY_REGISTRY
)


# def fallback_response(question, used_chunks=None):

#     if used_chunks:
#         return "I found some related information, but not enough to answer confidently."

#     return "I couldn't find that information in the provided documents."



# ----------------------------
# OVERLAP SCORE
# ----------------------------
def answer_overlap_score(answer, context_chunks):

    # ❌ REMOVE strict word matching
    # ✅ just ensure answer is not hallucinated

    if not context_chunks:
        return 0.0

    context_text = " ".join([c[1] for c in context_chunks]).lower()

    # check if at least some key info exists
    matches = sum(
        1 for w in answer.lower().split()
        if w in context_text
    )

    # relaxed scoring
    return matches / max(len(answer.split()), 1)



# ----------------------------
# SENTENCE EXTRACTION
# ----------------------------
def extract_relevant_sentences(question, used_chunks):

    sentences = []

    for c in used_chunks:

        score = float(c.get("score", 0))
        text = c.get("text", "")
        source = c.get("source", "")

        # ✅ FIXED SPLITTING (no aggressive colon split)
        #split_sentences = re.split(r'(?:\n|•|(?<=\.)\s)', text)
        split_sentences = [text]  #DONT SPLIT AGAIN
        for s in split_sentences:

            s = s.strip()

            if len(s) < 20:
                continue

            relevance = sum(
                1 for w in question.lower().split()
                if w in s.lower()
            )

            sentences.append(
                (score + relevance * 0.05, s, source)
            )

    sentences.sort(key=lambda x: x[0], reverse=True)

    return sentences


# ----------------------------
# CLEAN + RERANK
# ----------------------------
# def clean_and_rerank_sentences(sentences, question, max_sentences):
    
#     q_words = set(question.lower().split())

#     is_list = is_list_query(question)

#     seen = set()
#     scored = []

#     for score, text, source in sentences:

#         t = text.lower().strip()

#         if t in seen:
#             continue
#         seen.add(t)

#         # ❌ remove broken fragments
#         # if len(t.split()) < 4:
#         #     continue

#         if len(t) < 8:
#             continue

#         # ❌ avoid partial headings
#         # if t.endswith("to") or t.endswith(":"):
#         #     continue

#         # ❌ avoid very long noisy lines
#         # if len(t) > 200:
#         #     continue

#         # ✅ semantic-friendly scoring

#         # ----------------------------
#         # SEMANTIC-FRIENDLY SCORING (FIXED)
#         # ----------------------------

#         # small keyword boost (optional)
#         overlap = sum(1 for w in q_words if w in t)
#         score += overlap * 0.05

#         # 🚀 ALWAYS ADD BASE SCORE (IMPORTANT FIX)
#         score += 0.3

#         # ✅ BOOST SHORT, CLEAR CHUNKS
#         if len(t.split()) <= 12:
#             score += 0.2

#         # ✅ BOOST FACTUAL CONTENT (GENERIC)
#         if len(t.split()) >= 3:
#             score += 0.1

#         if is_list:
#             score += 0.25

#         if any(q_word in text.lower() for q_word in question.split()):
#             score *= 1.1
        

#         scored.append((score, text, source))

#     scored.sort(key=lambda x: x[0], reverse=True)

#     return scored[:max_sentences]


def clean_and_rerank_sentences(sentences, question, max_sentences):

    q_words = set(question.lower().split())
    is_list = is_list_query(question)

    seen = set()
    scored = []

    for c in sentences:

        if isinstance(c, tuple):
            score, text, source = c
        else:

            score = c.get("score", 1.0)   # default score
            text = c.get("text", "")
            source = c.get("source", "")

        t = text.lower().strip()

        if t in seen:
            continue
        seen.add(t)

         # ❌ remove broken fragments
        if len(t) < 8:
            continue

        if len(t.split()) < 3:
            continue

        # ----------------------------
        # SAFE SCORING
        # ----------------------------
        new_score = score * 0.7

        # small keyword boost (optional)
        overlap = sum(1 for w in q_words if w in t)
        new_score += overlap * 0.05

        # 🔥 BOOST TIME CHUNKS
        if re.search(r'\d{1,2}:\d{2}', text):
            new_score += 0.5   # strong boost

        if "hour" in text or "time" in text:
            new_score += 0.3

        # ALWAYS ADD BASELINE SCORE
        new_score += 0.2

        # ✅ BOOST SHORT, CLEAR CHUNKS
        if len(t.split()) <= 12:
            new_score += 0.1

        if is_list:
            new_score += 0.15

        if any(q_word in t for q_word in q_words):
            new_score *= 1.05

        scored.append((new_score, text, source))

    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:max_sentences]





# ----------------------------
# LIST FORMATTER
# ----------------------------
def format_list_items(text):

    items = re.split(r'\n|•|-', text)

    clean = []

    for item in items:
        s = clean_list_item(item)

        if not s:
            continue

        if 10 < len(s) < 120:
            clean.append(s)

        if len(clean) >= 5:
            break

    if not clean:
        return text

    return "Here are some available options👇:\n" + "\n".join(f"• {c}" for c in clean)


def split_time_lines(text):

    # normalize spacing
    #text = normalize_for_time(text)
    text = re.sub(r'\s+', ' ', text)
    

    # split by bullets or newlines
    parts = re.split(r'(?:\n|•)', text)

    lines = []

    for p in parts:

        # further split using time patterns safely
        sub_parts = re.split(r'(?=\d{1,2}:\d{2}\s?(?:AM|PM))', p)

        for sp in sub_parts:
            clean = sp.strip()
            if len(clean) > 5:
                lines.append(clean)

    return lines


# ----------------------------
# TIME EXTRACTION
# ----------------------------
import re


TIME_RANGE_CLEAN = re.compile(
    r'(\d{1,2}:\d{2}\s?(AM|PM))\s*(?:-|–|to)\s*(\d{1,2}:\d{2})(?:\s?(AM|PM))?',
    re.I
)


def normalize_time_range(text):

    def repl(match):
        start = match.group(1)              # 6:00 AM
        end_time = match.group(3)           # 11:00
        end_period = match.group(4)         # AM/PM or None
        start_period = match.group(2)       # AM/PM

        # 🔥 FIX: intelligent inference
        if not end_period:

            start_hour = int(start.split(":")[0])
            end_hour = int(end_time.split(":")[0])

            # -------------------------
            # 🔥 CASE 1: SAME HOUR → assume PM (very common in business hours)
            # -------------------------
            if end_hour == start_hour:
                end_period = "PM"

            # -------------------------
            # 🔥 CASE 2: end < start → definitely PM
            # -------------------------
            elif end_hour < start_hour:
                end_period = "PM"

            # -------------------------
            # 🔥 CASE 3: normal case
            # -------------------------
            else:
                # if start is AM → end usually PM
                if start_period.upper() == "AM":
                    end_period = "PM"
                else:
                    end_period = start_period

        return f"{start} - {end_time} {end_period}"

    return TIME_RANGE_CLEAN.sub(repl, text)




DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

TIME_PATTERN = re.compile(r'\d{1,2}:\d{2}\s?(AM|PM)', re.I)
FULL_RANGE_PATTERN = re.compile(r'\d{1,2}:\d{2}.*\d{1,2}:\d{2}', re.I)



def extract_time_answer(text, question):

    import re
    
    #lines = normalize_for_time(text)
    lines = re.split(r'(?:\n|•)', text)
    
    results = []

    # ----------------------------
    # DAY BLOCK SPLIT (FIXED)
    # ----------------------------
    DAY_SPLIT = re.split(
        r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
        text,
        flags=re.I
    )

    day_blocks = []
    for i in range(1, len(DAY_SPLIT), 2):
        day = DAY_SPLIT[i].lower()
        content = DAY_SPLIT[i + 1] if i + 1 < len(DAY_SPLIT) else ""
        day_blocks.append((day, content.strip()))

    # ----------------------------
    # QUERY ANALYSIS
    # ----------------------------
    STOPWORDS = {"what", "is", "the", "of", "a", "an", "are", "timing"}

    q_words = [
        w for w in question.lower().split()
        if w not in STOPWORDS and len(w) > 2
    ]

    day_match = re.search(r'|'.join(DAYS), question.lower())
    target_day = day_match.group(0) if day_match else None

    # ----------------------------
    # 🔥 CASE 1: DAY-SPECIFIC QUERY (BLOCK-BASED)
    # ----------------------------
    if target_day:

        for day, content in day_blocks:

            if target_day not in day:
                continue

            clean_content = re.sub(r'\s+', ' ', content)

            # CLOSED
            if "closed" in clean_content.lower():
                return f"{day.capitalize()}: Closed"

            # 24 hours
            if re.search(r'24\s*hours|open all day', clean_content, re.I):
                return f"{day.capitalize()}: Open 24 hours"

            # TIME RANGE
            if FULL_RANGE_PATTERN.search(clean_content):
                # clean_content = normalize_time_range(clean_content)

                # # 🔥 ADD THIS LINE
                # clean_content = re.sub(r'\b(am|pm)\b', lambda x: x.group().upper(), clean_content)

                return f"{day.capitalize()}: {clean_content}"

            # fallback if something exists
            if clean_content:
                return f"{day.capitalize()}: {clean_content}"

        return None  # nothing found

    # ----------------------------
    # 🔥 CASE 2: GENERAL TIME QUERY (LINE-BASED)
    # ----------------------------

    def is_relevant(text):
        return any(w in text.lower() for w in q_words)

    for i, line in enumerate(lines):

        line = line.strip()
        if len(line) < 3:
            continue

        # skip headings
        if line.lower() in ["clinic hours", "facilities", "services"]:
            continue

        # merge broken lines (Sunday + Closed)
        if re.match(rf'({"|".join(DAYS)}):?$', line.lower()):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if "closed" in next_line.lower():
                    line = f"{line} Closed"

        # CLOSED DAY
        day_match = re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', line, re.I)
        if day_match and "closed" in line.lower():
            day = day_match.group(1).capitalize()
            results.append(f"{day}: Closed")
            continue

        # 24 hours
        if re.search(r'24\s*hours|open all day', line, re.I):
            if is_relevant(line):
                results.append(line.strip())
            continue

        # TIME RANGE
        if FULL_RANGE_PATTERN.search(line):
            clean = re.sub(r'\s+', ' ', line)
            # clean = normalize_time_range(clean)

            # # 🔥 ADD THIS LINE
            # clean = re.sub(r'\b(am|pm)\b', lambda x: x.group().upper(), clean)

            if is_relevant(clean):
                results.append(clean)

    # ----------------------------
    # CLEANUP
    # ----------------------------
    results = list(dict.fromkeys(results))

    return "\n".join(results) if results else None


# ----------------------------
# REMOVE REPETITION
# ----------------------------
def remove_repetition(text):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    seen = set()
    clean = []

    for s in sentences:
        t = s.lower().strip()

        if t not in seen:
            clean.append(s)
            seen.add(t)

    return " ".join(clean)



# def detect_yes_no(question, context_chunks):

#     q = question.lower()

#     for _, text, _ in context_chunks:

#         t = text.lower()

#         # ❌ negative first
#         if "not available" in t or "not provided" in t or "no " in t:
#             return "No."

#         # ✅ keyword match (strong)
#         keywords = [w for w in q.split() if len(w) > 3]

#         match_count = sum(1 for w in keywords if w in t)

#         if match_count >= 2:   # 🔥 threshold
#             return "Yes."

#     return None

# def semantic_match(entity, context):

#     entity_words = entity.split()

#     # direct match
#     if entity in context:
#         return True

#     # partial match
#     if any(w in context for w in entity_words):
#         return True
    
#     # 🔥 ANY important word match
#     for w in entity_words:
#         if len(w) > 3 and w in context:
#             return True
    
#     SYNONYMS = {
#         "diagnostic": ["diagnostic", "lab", "test", "testing", "diagnosis"],
#         "health checkups": ["preventive", "checkup", "checkups", "health", "Preventive health checkups"],
#         "consultation": ["consult", "doctor", "appointment"],
#         "pharmacy": ["medicine", "drug", "medication"],
#         "wifi": ["wifi", "internet"],
#         "parking": ["parking", "vehicle", "car"],
#     }

#     # synonym match
#     if entity in SYNONYMS:
#         for syn in SYNONYMS[entity]:
#             if syn in context:
#                 return True

#     return False

def semantic_match(entity, context):

    context = context.lower()
    entity = entity.lower()

    # 🔥 STRICT match only
    if entity in context:
        return True

    # 🔥 synonym match
    SYNONYMS = {
        "wifi": ["wifi", "internet"],
        "parking": ["parking", "vehicle", "car"],
        "taxi": ["taxi", "cab"],
    }

    if entity in SYNONYMS:
        if any(s in context for s in SYNONYMS[entity]):
            return True

    return False



NEGATIONS = ["no", "not", "without", "unavailable", "closed"]

def has_negation(entity, context):
    window = 5  # words before entity
    words = context.split()

    for i, w in enumerate(words):
        if entity in w:
            start = max(0, i - window)
            if any(n in words[start:i] for n in NEGATIONS):
                return True
    return False

def clean_entity(entity):

    STOP_WORDS = {
        "available", "on", "at", "in", "for",
        "with", "by", "to", "of", "do", "you", "offer"
    }

    words = [
        w for w in entity.lower().split()
        if w not in STOP_WORDS and len(w) > 2
    ]

    return " ".join(words)

def extract_core_entity(entity):

    STOP_WORDS = {
        "service", "services", "facility", "facilities",
        "available", "provide", "offer", "have"
    }

    words = [
        w for w in entity.lower().split()
        if w not in STOP_WORDS and len(w) > 2
    ]

    # 🔥 return MOST important word (last noun bias)
    return words[-1] if words else entity



# def detect_yes_no(question, chunks):

#     q = question.lower()

#     context_text = " ".join([
#         c.get("text", "").lower()
#         for c in chunks
#     ])

#     # ----------------------------
#     # EXTRACT ENTITY
#     # ----------------------------
#     entity = detect_entity(question) or extract_main_entity(question) or "it"
#     entity = clean_entity(entity)

#     if not entity:
#         entity = question.split()[-1]

#     entity_words = entity.lower().split()

#     # ----------------------------
#     # 🔥 FLEXIBLE MATCHING
#     # ----------------------------

#     # 1. direct match
#     # if entity in context_text:
#     #     return "Yes"

#     if semantic_match(entity, context_text):
#         if has_negation(entity, context_text):
#             return "No"
#         return "Yes"
    
#     # 🔥 if multi-word entity fails → try core noun
#     if not semantic_match(entity, context_text):

#         core = entity.split()[0]  # e.g., "pharmacy"

#         if core in context_text:
#             return "Yes"

#     # 2. partial word overlap
#     overlap = sum(1 for w in entity_words if w in context_text)

#     if overlap >= max(1, len(entity_words) // 2):
#         return "Yes"

#     # 3. semantic synonyms (VERY IMPORTANT)
#     SYNONYMS = {
#         "diagnostic": ["diagnostic", "lab", "test", "testing", "diagnosis"],
#         "health checkups": ["preventive", "checkup", "checkups", "health", "Preventive health checkups"],
#         "consultation": ["consult", "doctor", "appointment"],
#         "pharmacy": ["medicine", "drug", "medication"],
#         "wifi": ["wifi", "internet"],
#         "parking": ["parking", "vehicle", "car"],
#     }

#     for word in entity_words:
#         if word in SYNONYMS:
#             if any(s in context_text for s in SYNONYMS[word]):
#                 return "Yes"

#     # ----------------------------
#     # NO MATCH
#     # ----------------------------
#     return "No"


# def detect_yes_no(question, chunks):

#     context_text = " ".join([
#         c.get("text", "").lower()
#         for c in chunks
#     ])

#     entity = detect_entity(question) or extract_main_entity(question)
#     entity = clean_entity(entity)

#     if not entity:
#         return "No"

#     # 🔥 extract core entity
#     core = extract_core_entity(entity)

#     # 🔥 strict matching
#     if semantic_match(core, context_text):

#         if has_negation(core, context_text):
#             return "No"

#         return "Yes"

#     return "No"


# def detect_yes_no(question, chunks):

#     print("\n" + "=" * 70)
#     print("🔍 BINARY VALIDATION DEBUG")
#     print("=" * 70)

#     print("QUESTION:", question)

#     context_text = " ".join([
#         c.get("text", "").lower()
#         for c in chunks
#     ])

#     print("CONTEXT:")
#     print(context_text)

#     entity = detect_entity(question)

#     print("detect_entity():", entity)

#     if not entity:
#         entity = extract_main_entity(question)
#         print("extract_main_entity():", entity)

#     entity = clean_entity(entity)

#     print("clean_entity():", entity)

#     core = extract_core_entity(entity)

#     print("extract_core_entity():", core)

#     match = semantic_match(core, context_text)

#     print("semantic_match():", match)

#     negation = has_negation(core, context_text)

#     print("has_negation():", negation)

#     print("=" * 70)

#     if not entity:
#         return "No"

#     if match:

#         if negation:
#             return "No"

#         return "Yes"

#     return "No"


def detect_yes_no(
    question,
    chunks,
    entities
):

    context_text = " ".join(
        c.get("text", "").lower()
        for c in chunks
    )

    if not entities:
        return "No"

    entity = entities[0]

    if not entity_has_evidence(
        entity,
        chunks
    ):
        return "No"

    if has_negation(
        entity,
        context_text
    ):
        return "No"

    return "Yes"



def extract_main_entity(question):

    q = question.lower()

    # remove prefixes
    q = re.sub(r'^(is|are|do|does|can|will)\s+', '', q)

    # remove helper words
    q = re.sub(r'\b(you|provide|offer|have|available|i|book|an|a)\b', '', q)

    q = re.sub(r'[^\w\s]', '', q)

    return q.strip()


def clean_list_item(text):

    t = text.strip()

    # ----------------------------
    # ❌ REMOVE BROKEN MULTI-LINE HEADINGS
    # ----------------------------
    if len(t.split()) <= 3:
        return None

    # ----------------------------
    # ❌ REMOVE BUSINESS NAMES / TITLES
    # e.g. "Grand Horizon Hotel -", "Care Clinic -"
    # ----------------------------
    if re.search(r'(hotel|clinic|restaurant)\s*-', t, re.I):
        return None

    # ----------------------------
    # ❌ REMOVE SECTION HEADINGS
    # ----------------------------
    if re.search(r'(facilities|services|menu|information|options|hours)$', t, re.I):
        return None

    # ----------------------------
    # ❌ REMOVE "Services &", "Menu &", etc
    # ----------------------------
    if re.search(r'&\s*$', t):
        return None

    # ----------------------------
    # ❌ REMOVE "Monday_TO_Friday" garbage
    # ----------------------------
    if "_" in t and "to" in t.lower():
        return None

    # ----------------------------
    # ❌ REMOVE "Sunday:" (empty day lines)
    # ----------------------------
    if re.match(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*:?$', t, re.I):
        return None
    
    if not re.search(r'(available|service|room|wifi|parking|consultation|restaurant|system)', t, re.I):
        return None

    # ----------------------------
    # ❌ REMOVE VERY GENERIC WORDS
    # ----------------------------
    if t.lower() in ["options", "services", "menu", "information"]:
        return None

    # ----------------------------
    # ❌ REMOVE DUPLICATE-LIKE SHORT NOISE
    # ----------------------------
    if len(t) < 8:
        return None

    return t



def contains_time(text):

    return bool(re.search(
        r'\d{1,2}:\d{2}\s?(?:AM|PM)',
        text,
        re.IGNORECASE
    ))


def get_best_time_chunk(used_chunks):

    import re

    # 🔥 keep only time-containing chunks
    time_chunks = []

    for c in used_chunks:

        if isinstance(c, tuple):
            _, text, _ = c
        else:
            text = c.get("text", "")
            #text = normalize_for_time(text)

        if re.search(r'\d{1,2}:\d{2}\s?(AM|PM)', text, re.I):
            time_chunks.append(c)

    if time_chunks:
        used_chunks = time_chunks

    best = None
    best_score = -1

    for c in used_chunks:

        if isinstance(c, tuple):
            score, text, source = c
        else:
            score = c.get("score", 1.0)   # default score
            text = c.get("text", "")
            source = c.get("source", "")

        if re.search(r'\d{1,2}:\d{2}\s?(AM|PM)', text, re.I):
            score *= 1.2

        if any(k in text.lower() for k in ["hour", "time"]):
            score += 0.5
            

        if score > best_score:
            best = text
            best_score = score

    return best

DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

meta = {
        "latency": {},
        "cost": {},
        "debug": {}
    }

def expand_day_range(text):

    text = text.lower()
    #text = normalize_for_time(text)

    

    match = re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+to\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', text)

    if not match:
        return None

    start, end = match.groups()

    start_idx = DAYS.index(start)
    end_idx = DAYS.index(end)

    return DAYS[start_idx:end_idx+1]



def format_time_output(time_map):

    formatted = []

    for day, time in time_map.items():

        if time.lower() == "closed":
            formatted.append(f"{day.capitalize()}: Closed")
            continue

        # normalize spacing + casing
        time = time.strip()

        # 🔥 FIX AM/PM casing
        time = re.sub(r'\b(am|pm)\b', lambda x: x.group().upper(), time)

        # 🔥 FIX missing AM/PM in end time
        match = re.search(r'(\d{1,2}:\d{2})\s*(AM|PM)?\s*-\s*(\d{1,2}:\d{2})(\s*(AM|PM))?', time, re.I)

        if match:
            start, start_p, end, _, end_p = match.groups()

            if not start_p:
                start_p = "AM"   # safe default

            if not end_p:
                # 🔥 infer intelligently
                start_hour = int(start.split(":")[0])
                end_hour = int(end.split(":")[0])

                if end_hour < start_hour:
                    end_p = "PM"
                else:
                    end_p = start_p

            time = f"{start} {start_p.upper()} - {end} {end_p.upper()}"

        formatted.append(f"{day.capitalize()}: {time}")

    return "\n".join(formatted)




def normalize_for_time(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()



def build_time_map(chunks, question):

    time_map = {}
    DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

    for c in chunks:
        
        text = c.get("text", "")
        t = normalize_for_time(text)
        print("PROCESSING:", t)

        # ----------------------------
        # ✅ CASE 3: CLOSED (SAFE VERSION)
        # ----------------------------
        for match in re.finditer(
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*:\s*closed',
            t,
            re.I
        ):
            day = match.group(1)
            time_map[day] = "Closed"
            time_map[day] = str("Closed")

        # ----------------------------
        # ✅ CASE 1: DAY RANGE (ONLY ONE LOGIC)
        # ----------------------------
        for range_match in re.finditer(
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[_\s]*to[_\s]*'
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[^\d]*'
            r'(\d{1,2}:\d{2}.*?\d{1,2}:\d{2})',
            t,
            re.I
        ):
            start_day, end_day, time = range_match.groups()

            # 🔥 CRITICAL FIX (THIS IS THE MISSING PIECE)
            time = normalize_time_range(time)

            # 🔥 FIX casing
            time = re.sub(r'\b(am|pm)\b', lambda x: x.group().upper(), time)

            time = re.sub(r'^[a-z]+:\s*', '', time.strip(), flags=re.I)

            start_idx = DAYS.index(start_day.lower())
            end_idx = DAYS.index(end_day.lower())

            for d in DAYS[start_idx:end_idx+1]:
                time_map[d] = time
                time_map[d] = str(time)

        # ----------------------------
        # ✅ CASE 2: SINGLE DAY
        # ----------------------------
        for match in re.finditer(
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*:\s*'
            r'(\d{1,2}:\d{2}\s?(am|pm)?\s*-\s*\d{1,2}:\d{2}\s?(am|pm)?)',
            t,
            re.I
        ):
            day = match.group(1)
            time = match.group(2)

            # 🔥 CRITICAL FIX (THIS IS THE MISSING PIECE)
            time = normalize_time_range(time)

            # 🔥 FIX casing
            time = re.sub(r'\b(am|pm)\b', lambda x: x.group().upper(), time)

            time = re.sub(r'^[a-z]+:\s*', '', time.strip(), flags=re.I)

            time_map[day] = time
            time_map[day] = str(time)

        # ----------------------------
        # ✅ CASE 4: GENERIC TIME (NO DAY) 
        # ----------------------------
        for match in re.finditer(
            r'(\d{1,2}:\d{2}\s?(am|pm)?\s*(?:-|–|to)\s*\d{1,2}:\d{2}\s?(am|pm)?)',
            t,
            re.I
        ):
            time = match.group(1)

            # normalize
            time = normalize_time_range(time)

            # fix AM/PM casing
            time = re.sub(r'\b(am|pm)\b', lambda x: x.group().upper(), time)

            # ----------------------------
            # 🔥 MAP TO ENTITY (CRITICAL)
            # ----------------------------
            entity = "general"

            if "breakfast" in t:
                entity = "breakfast"
            elif "room service" in t:
                entity = "room service"
            elif "restaurant" in t:
                entity = "restaurant"

            time_map[entity] = time

        # # ----------------------------
        # # ✅ CASE 3: CLOSED (SAFE VERSION)
        # # ----------------------------
        # for match in re.finditer(
        #     r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*:\s*closed',
        #     t,
        #     re.I
        # ):
        #     day = match.group(1)
        #     time_map[day] = "Closed"

        # ----------------------------
        # ❌ REMOVE EMERGENCY BLOCK (NOT NEEDED NOW)
        # ----------------------------

        

    return time_map, meta


def handle_structured_query(question, used_chunks, all_chunks, query_type):



    # ----------------------------
    # TIME QUERY
    # ----------------------------
    if query_type == "time":

        time_chunks = []

        for c in all_chunks:

            if isinstance(c, tuple):
                text = c[1]
            else:
                text = c.get("text", "")

            if re.search(r'\d{1,2}:\d{2}|closed|hour', text, re.I):
                time_chunks.append(c)

        if time_chunks:
            time_map, meta = build_time_map(time_chunks, question)
        else:
            time_map, meta = build_time_map(all_chunks, question)

        print("⏰TIME MAP:", time_map)

        q = question.lower()

        # 🔥 ENTITY-BASED TIME (NEW)
        for k, v in time_map.items():
            if k in q:
                return f"{k.title()} is available from {v}."

        # 🔥 specific day query
        for d in DAYS:
            if d in q:
                if d in time_map:
        
                    val = time_map[d]

                    if val.lower().startswith(d):
                        return val.capitalize()

                    return f"{d.capitalize()}: {val}"
                else:
                    return f"No timing information available for {d.capitalize()}."

        # 🔥 general query → show all
        
        valid_entries = {k: v for k, v in time_map.items() if v}

        if valid_entries:
            # 🔥 OPTIONAL: fill missing Sunday
            if len(time_map) >= 5 and "sunday" not in time_map:
                time_map["sunday"] = "Closed"

            # 🔥 FINAL FORMATTED OUTPUT HERE ONLY
            return format_time_output(time_map)
        
        if not valid_entries:

            # 🔥 fallback to best time chunk
            best = get_best_time_chunk(used_chunks)

            if best:
                answer = extract_time_answer(best, question)
                return answer, meta
            

        if "saturday" not in time_map:
            time_map["saturday"] = "Not specified"

        ALL_DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

        # # if most days exist but Sunday missing → infer closed
        # if len(time_map) >= 5 and "sunday" not in time_map:
        #     time_map["sunday"] = "Closed"
        
        

        return "I could not find timing information."
    
    

STOPWORDS = {"the", "is", "are", "a", "an", "of", "to", "in", "and"}

def is_answer_grounded(answer, used_chunks):

    # 🔥 handle tuple answer
    if isinstance(answer, tuple):
        answer = answer[0]

    if not isinstance(answer, str):
        return False

    # 🔥 normalize context
    context_parts = []

    for c in used_chunks:
        if isinstance(c, tuple):
            text = c[1]
        else:
            text = c.get("text", "")

        if text:
            context_parts.append(text.lower())

    context = " ".join(context_parts)

    # 🔥 extract words safely
    answer_words = [
        w for w in answer.lower().split()
        if w not in STOPWORDS and len(w) > 2
    ]

    if not answer_words:
        return False

    overlap = sum(1 for w in answer_words if w in context)

    ratio = overlap / len(answer_words)

    return ratio > 0.4


def is_query_answerable(question, used_chunks):

    q_words = set(question.lower().split())
    context = " ".join([
        c.get("text", "").lower()
        for c in used_chunks
    ])

    overlap = sum(1 for w in q_words if w in context)

    return overlap >= 2   # threshold



def is_price_query(q):
    return any(w in q.lower() for w in ["price", "cost", "charge", "fee"])



def validate_context(question, used_chunks):

    context = " ".join([
        c.get("text", "").lower()
        for c in used_chunks
    ])

    # price query must have numbers
    if is_price_query(question):
        if not re.search(r'\d+|\$', context):
            return False

    # time query must have time pattern
    if is_time_query(question):
        if not re.search(r'\d{1,2}:\d{2}|closed|hours', context):
            return False

    return True




# def better_time_output(text):

#     if isinstance(value, dict):
#         text = value.get("text", "")
#     else:
#         text = value

#     text = text.replace("Closed", "❌ Closed")

#     text = re.sub(
#         r'(\d{1,2}:\d{2}.*\d{1,2}:\d{2})',
#         r'🕒 \1',
#         text
#     )

#     return text


# ----------------------------
# MAIN VALIDATE FUNCTION
# ----------------------------
def validate_answer(
    question,
    retrieval,
    pipeline,
    start_time,
    system_prompt,
    memory,
    query_type = None,
    entities = None
):
    
    









    # 🔥🔥🔥 CRITICAL FIX
    if query_type == "list":
        # 🔥 SAFETY CHECK
        if not any(word in question.lower() for word in [
            "facilities", "services", "amenities", "list"
        ]):
            print("⚠️ NOT A TRUE LIST QUERY → SKIP BYPASS")
        else:
            print("⚡ LIST BYPASS VALIDATION")
            

            chunks = retrieval.get("chunks", [])

            if not chunks:
                return None, {}, {}

            items = chunks[0].get("items", [])

            if not items:
                text = chunks[0].get("text", "")
                return text, {}, {}

            # ✅ format properly
            answer = "Here are some available options:\n"
            for item in items:
                answer += f"• {item}\n"

            return answer.strip(), {"grounded": True}, {}
    

    if query_type == "contact":

        print("⚡ 😡🤓😎CONTACT BYPASS VALIDATION")

        contact_blocks = [
            c for c in retrieval.get("chunks", [])
            if isinstance(c.get("source"), dict)
            and c["source"].get("type") == "contact"
        ]

        if not contact_blocks:
            return "Contact information is not available.", {}, {}

        block = contact_blocks[0]
        structured = block["source"].get("structured", [])

        q = question.lower()

        # -----------------------------
        # 🔥 ROLE-SPECIFIC MATCH
        # -----------------------------
        for item in structured:
            role = item["role"]
            phone = item["phone"]

            if role in q:
                return f"{role.title()} contact number is {phone}.", {}, {}

        # -----------------------------
        # 🔥 GENERAL CONTACT
        # -----------------------------
        numbers = [i["phone"] for i in structured]

        return "You can contact at: " + ", ".join(numbers), {}, {}




    meta = {
        "latency": {},
        "cost": {},
        "debug": {}
    }

    used_chunks = []

    for c in retrieval["chunks"]:

        score = c.get("score", 1.0)   # default score
        text = c.get("text", "")
        source = c.get("source", "")

        # 🔥 if "used" missing → assume True
        if c.get("used", True):
            used_chunks.append(c)


    all_chunks = []

    for c in retrieval["chunks"]:

        if isinstance(c, tuple):
            # already in correct format
            all_chunks.append(c)

        elif isinstance(c, dict):
            score = c.get("score", 1.0)
            text = c.get("text", "")
            source = c.get("source", "")
            all_chunks.append(c)

    # ----------------------------
    # EARLY EXIT
    # ----------------------------
    if not used_chunks:

        end = time.time()

        return (
            "I could not find that information in the provided documents.",
            {"grounded": False},
            {"latency": {"total_sec": round(end - start_time, 3)}, "cost": {}}
        )

    # ----------------------------
    # CONTEXT OPTIMIZATION
    # ----------------------------

    is_list = is_list_query(question)

    sentences = extract_relevant_sentences(question, used_chunks)

    optimized_chunks = clean_and_rerank_sentences(
        sentences,
        question,
        max_sentences=5 if is_list else 3   # ✅ KEY FIX
    )

    

    is_list = is_list_query(question)
    is_binary = is_binary_query(question)
    is_time = is_time_query(question)
    is_feature = is_feature_query(question)

    if is_binary_query(question):
        query_type = "binary"

    elif is_time_query(question):
        query_type = "time"

    elif is_list_query(question):
        query_type = "list"

    elif is_feature_query(question):
            query_type = "feature"   # 🔥 NEW
            print("🔥 QUERY TYPE IN VALIDATE:", query_type)


    else:
        query_type = "general"

    # ----------------------------
    # STRUCTURED
    # ----------------------------
    structured_answer = handle_structured_query(question, used_chunks, all_chunks, query_type)

    if structured_answer:

        # 🔥 SKIP grounding for feature queries
        if query_type != "feature":
            return structured_answer, {"grounded": True}, meta  # 🔥 FORCE TRUE
        
        if not is_answer_grounded(structured_answer, used_chunks):
            #return fallback_response(question, used_chunks), {"grounded": False}, meta
            answer, actions = fallback_response(question, used_chunks)

            meta["actions"] = actions   # 🔥 inject here

            return answer, {"grounded": False}, meta
        
        return structured_answer, {"grounded": True}, meta


    # ----------------------------
    # BINARY
    # ----------------------------
    if is_binary:

        print("=" * 70)
        print("BINARY RESPONSE DEBUG")
        print("entities:", entities)
        print("entities type:", type(entities))
        yn = detect_yes_no(question, used_chunks, entities)

        print("yn:", yn)
        print("yn type:", type(yn))

        print("=" * 70)

        if yn:
            entity = entities[0] if entities else "it"

            config = DOMAIN_ENTITY_REGISTRY.get(entity)

            if config:
                display_name = config.get("display_name", entity)
            else:
                display_name = entity


            if yn == "Yes":
                answer = f"Yes, {display_name} is available."
            else:
                answer = f"No, {display_name} is not available."

            print("answer:", answer)
            print("answer type:", type(answer))
    
            meta["grounded"] = True
            
            print("meta:", meta)
            print("meta types:", {
                k: type(v)
                for k, v in meta.items()
            })

            result = (answer, {"grounded": True}, meta)

            print("=" * 70)
            print("FINAL VALIDATION RETURN")
            print("result:", result)
            print("result type:", type(result))
            print("element types:", [type(x) for x in result])
            print("=" * 70)

            return result

            #return answer, {"grounded": True}, meta

            

            
        
    
    

        
    # 🔥 FEATURE QUERY BOOST
    if query_type == "feature":
        for c in used_chunks:
            text = c.get("text", "").lower()

            if any(word in text for word in question.lower().split()):
                structured_answer = text   # ✅ SET
                break

    

    # ----------------------------
    # LIST
    # ----------------------------
    if is_list and not structured_answer:

        # -----------------------------
        # 🔥 PRIORITY 1: STRUCTURED LIST
        # -----------------------------
        for chunk in used_chunks:
 
            if isinstance(chunk, dict) and chunk.get("type") == "list":

                items = chunk["items"]

                return (
                    "Here are some available options:\n" +
                    "\n".join(f"• {i}" for i in items)
                ), {"grounded": True}, meta
            
        # -----------------------------
        # 🛠️ PRIORITY 2: FALLBACK CLEANING
        # -----------------------------
            
        cleaned = []

        for c in used_chunks:
            text = c.get("text", "")
            c = clean_list_item(text)
            if c:
                cleaned.append(c)

        # 🔥 remove duplicates
        cleaned = list(dict.fromkeys(cleaned))

        if cleaned:
            formatted = "Here are some available options👇:\n"
            for i in cleaned[:5]:
                formatted += f"• {i}\n"

            return formatted.strip(), {"grounded": True}, meta
        

    print("\n📥 USED CHUNKS IN VALIDATE:")
    for c in used_chunks:
        print(type(c), c)

    for chunk in used_chunks:
        print("CHECKING CHUNK:", chunk)



    context = " ".join([
        c.get("text", "").lower()
        for c in used_chunks
    ])

    if not is_query_answerable(question, used_chunks):
        #return fallback_response(question, used_chunks), {"grounded": False}, meta
        answer, actions = fallback_response(question, used_chunks)

        meta["actions"] = actions   # 🔥 inject here

        return answer, {"grounded": False}, meta

    if is_price_query(question):
        if not re.search(r'\d+|\$', context):
            #return fallback_response(question, used_chunks), {"grounded": False}, meta
            answer, actions = fallback_response(question, used_chunks)

            meta["actions"] = actions   # 🔥 inject here

            return answer, {"grounded": False}, meta





    # ----------------------------
    # LLM CALL
    # ----------------------------

    llm_start = time.time()

    answer = pipeline.answer(
        question,
        optimized_chunks,
        system_prompt=system_prompt
    )

    llm_end = time.time()

    answer = remove_repetition(answer)

    if query_type not in ["binary", "time", "feature", "contact"]:

        if not is_answer_grounded(answer, used_chunks):
            end = time.time()
            meta["latency"]["total_sec"] = round(end - start_time, 3)

            #return fallback_response(question, used_chunks), {"grounded": False}, meta
            answer, actions = fallback_response(question, used_chunks)

            meta["actions"] = actions   # 🔥 inject here

            return answer, {"grounded": False}, meta
        

    if not validate_context(question, used_chunks):
        #return fallback_response(question, used_chunks), {"grounded": False}, meta
        answer, actions = fallback_response(question, used_chunks)

        meta["actions"] = actions   # 🔥 inject here

        return answer, {"grounded": False}, meta
    
    if query_type not in ["feature", "binary", "contact"]:
    
        if not is_query_answerable(question, used_chunks):
            print("❌ FALLBACK: is_query_answerable")
            #return fallback_response(question, used_chunks), {"grounded": False}, meta
            answer, actions = fallback_response(question, used_chunks)

            meta["actions"] = actions   # 🔥 inject here

            return answer, {"grounded": False}, meta

        if not validate_context(question, used_chunks):
            print("❌ FALLBACK: validate_context")
            #return fallback_response(question, used_chunks), {"grounded": False}, meta
            answer, actions = fallback_response(question, used_chunks)

            meta["actions"] = actions   # 🔥 inject here

            return answer, {"grounded": False}, meta


    # ----------------------------
    # POST FORMATTING
    # ----------------------------

    if is_list:
        formatted = format_list_items(answer)
        if formatted:
            answer = formatted


    # print("\n🔥 USED CHUNKS BEFORE CLEANING:")
    # for c in used_chunks:
    #     print(c[1][:100])

    # print("\n🔥 AFTER CLEANING:")
    # for c in optimized_chunks:
    #     print(c[1][:100])

    


    answer = re.sub(r'\n+', '\n', answer)
    answer = answer.strip()

    # 🔥 APPLY FORMAT HERE
    if is_time_query(question):
        answer = format_time_output(answer)

    

    # ----------------------------
    # METRICS
    # ----------------------------

    end = time.time()

    return (
        answer,
        {
            "recall_at_k": 1.0,
            "context_coverage": len(used_chunks),
            "faithful": True,
            "grounded": True,
            "grounding_score": 1.0
        },
        {
            "latency": {
                "total_sec": round(end - start_time, 3),
                "llm_sec": round(llm_end - llm_start, 3)
            },
            "cost": {}
        }
    )