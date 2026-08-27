import re
from agent.utils.query_classifier import resolve_entity_from_text, DOMAIN_ENTITY_REGISTRY

def clean_text(text):

    text = re.sub(r"\(cid:\d+\)", "", text)

    # normalize dashes FIRST
    text = text.replace("–", "-").replace("—", "-")

    # normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def protect_time(text):

    return re.sub(
        r'(\d{1,2}:\d{2}\s?(AM|PM))',
        r'__TIME__\1__TIME__',
        text,
        flags=re.I
    )


# ----------------------------------------
# NEW: SECTION SPLITTING (IMPORTANT 🔥)
# ----------------------------------------
# def split_into_sections(text):

#     # split when a heading-like pattern appears
#     sections = re.split(
#         r'(?=\n?[A-Z][A-Za-z\s]{3,}\n)',
#         text
#     )

#     return [s.strip() for s in sections if len(s.strip()) > 30]

def split_into_sections(text):

    sections = re.split(
        #r'(?=\n?[A-Z][A-Za-z\s&]{3,}\n)',
        r'(?m)(?=^[ \t]*[A-Z][A-Za-z&]*(?:[ \t]+[A-Za-z&]+)*[ \t]*$)',
        text
    )

    return [
        s.strip()
        for s in sections
        if s.strip()
    ]


def fix_time_fragments(text):

    text = re.sub(
        r'(\d{1,2}:\d{2}\s?(AM|PM))\s*-\s*\n\s*(\d{1,2}:\d{2}\s?(AM|PM))',
        r'\1 - \3',
        text,
        flags=re.I
    )


    return text



# ----------------------------------------
# FALLBACK: sentence splitting (keep yours)
# # ----------------------------------------
# def split_into_sentences(text):

#     sentences = re.split(
#         r'(?:\n+|•)',   # 🔥 REMOVE "-\s+"
#         text
#     )
#     # sentences = re.split(
#     #     r'(?:\n+|•|\.\s+(?=[A-Z]))',
#     #     text
#     # )

#     return [s.strip() for s in sentences if len(s.strip()) > 10]

# def split_into_sentences(text):
#     sentences = re.split(
#         r'(?:\n{2,})',  # 🔥 split only on double newline
#         text
#     )

#     return [s.strip() for s in sentences if len(s.strip()) > 20]

# def split_into_sentences(text):
#     # 🔥 KEEP SENTENCES INTACT
#     sentences = text.split("\n")

#     cleaned = []

#     for s in sentences:
#         s = s.strip()

#         # ❌ DO NOT DROP SHORT LINES
#         if len(s) < 3:
#             continue

#         cleaned.append(s)

# #     return cleaned

# def split_into_sentences(text):

#     sentences = re.split(r'\r?\n+', text)

#     return [
#         s.strip()
#         for s in sentences
#         if len(s.strip()) >= 3
#     ]

def split_into_sentences(text):

    print("\n" + "=" * 70)
    print("INPUT TO split_into_sentences")
    print(repr(text))
    print("=" * 70)

    sentences = text.split("\n")

    print("RAW SPLIT:")
    for i, s in enumerate(sentences):
        print(i, repr(s))

    cleaned = []

    for s in sentences:

        s = s.strip()

        if len(s) < 3:
            continue

        cleaned.append(s)

    print("\nFINAL SENTENCES:")
    for i, s in enumerate(cleaned):
        print(i, repr(s))

    return cleaned



SECTION_PATTERNS = [
    "facilities",
    "services",
    "amenities",
    "dining",
    "restaurant",
    "breakfast",
    "menu",
    "parking",
    "internet",
    "wifi",
    "rooms",
    "policies",
    "contact",
    "pricing",
    "appointments",
    "hours",
    "timing",
    "clinic"
]


def detect_section(text):

    text_lower = text.lower()

    for section in SECTION_PATTERNS:
        if section in text_lower:
            return section

    return "general"


def restore_structure_for_lists(text):
    return re.sub(
        r'(?<!^)(?=[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        '\n',
        text
    )



# def clean_chunk_text(text):

#     lines = re.split(r'(?:\n|•)', text)

#     clean_lines = []

#     for line in lines:

#         line = line.strip()

#         if len(line) < 5:
#             continue

#         # remove broken headings
#         if line.endswith("("):
#             continue

#         if len(line.split()) <= 2:
#             continue

#         # ❌ remove broken headings / fragments
#         if re.match(r'^[A-Za-z\s\-&()]+$', line) and len(line.split()) <= 3:
#             continue

#         # ❌ remove words like "Grill", "Options"
#         if len(line.split()) == 1:
#             continue

#         merged_lines = []
#         buffer = ""

#         for line in lines:

#             line = line.strip()

#             if not line:
#                 continue

#             # 🔥 if previous line incomplete → merge
#             if buffer and not buffer.endswith(('.', ':')):
#                 buffer += " " + line
#             else:
#                 if buffer:
#                     merged_lines.append(buffer)
#                 buffer = line

#         if buffer:
#             merged_lines.append(buffer)

#         lines = merged_lines

#         # if len(line.split()) < 3:
#         #     continue

#         # normalize dash
#         line = line.replace("–", "-").replace("—", "-")

#         clean_lines.append(line)

#     return clean_lines    # return LIST instead of string Because we'll further use t = text.lower().replace("\n", " ") in build_time_map function


def clean_chunk_text(text):

    lines = re.split(r'(?:\n|•)', text)

    clean_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove extremely short fragments only
        if len(line) < 3:
            continue

        # Remove obviously broken heading fragments
        if line.endswith("("):
            continue

        # Normalize dashes
        line = (
            line
            .replace("–", "-")
            .replace("—", "-")
        )

        clean_lines.append(line)

    return clean_lines

def parse_contact_items(items):
    parsed = []

    for line in items:
        parts = line.split(":")

        if len(parts) == 2:
            role = parts[0].strip().lower()
            phone = parts[1].strip()

            parsed.append({
                "role": role,
                "phone": phone
            })

    return parsed


def is_contact_heading(line):
    return "contact" in line.lower() or "phone" in line.lower()

import re

def extract_contact_items(lines, start_index):

    contacts = []

    for i in range(start_index + 1, min(start_index + 5, len(lines))):
        line = lines[i]

        # 🔥 phone number pattern
        numbers = re.findall(r'\b\d{10}\b', line)

        if numbers:
            contacts.append(line)

        else:
            break  # stop when pattern breaks

    return contacts

def extract_phone_numbers(text):
    return re.findall(r'\b\d{10}\b', text)


# def is_heading(line):

#     line = line.strip()

#     # ❌ reject time lines
#     if re.search(r'\d{1,2}:\d{2}', line):
#         return False

#     # ❌ reject very long lines
#     if len(line.split()) > 5:
#         return False

#     # ✅ allow title-like phrases
#     if line.istitle():   # <-- KEY FIX
#         return True

#     # ✅ fallback: contains keywords
#     if any(k in line.lower() for k in [
#         "dishes", "menu", "services", "facilities",
#         "hours", "beverages", "options"
#     ]):
#         return True

#     return False

KNOWN_HEADINGS = [
    "facilities",
    "breakfast",
    "room service",
    "meeting rooms",
    "additional services"
]


# def is_heading(line):
#     line = line.strip().lower()

#     known_headings = [
#         "facilities",
#         "breakfast",
#         "room service",
#         "meeting room",
#         "additional services"
#     ]

#     # ✅ exact match (strongest)
#     if line in known_headings:
#         return True

#     # ❌ reject time lines
#     if re.search(r'\d{1,2}:\d{2}', line):
#         return False

#     # ❌ reject sentences
#     if any(word in line for word in [
#         "available", "included", "service", "free", "access"
#     ]):
#         return False

#     # ✅ short phrase heuristic
#     if len(line.split()) <= 3:
#         return True

#     return False

def is_heading(line):
    line_clean = line.strip()
    line_lower = line_clean.lower()

    # 🔥 1. reject empty
    if not line_clean:
        return False

    # 🔥 2. reject time patterns
    if re.search(r'\d{1,2}:\d{2}', line_clean):
        return False

    # 🔥 3. reject long sentences
    if len(line_clean.split()) > 6:
        return False

    # 🔥 4. reject clear descriptive lines
    if any(word in line_lower for word in [
        "available", "included", "with", "and", "for", "to"
    ]):
        return False

    # 🔥 5. strong signal: title case or uppercase
    if line_clean.istitle() or line_clean.isupper():
        return True

    # 🔥 6. short phrase (BUT safe)
    if 1 <= len(line_clean.split()) <= 3:
        return True

    return False

def is_real_heading(lines, i):
    line = lines[i]

    if not is_heading(line):
        return False

    # 🔥 LOOK AHEAD
    next_items = 0

    for j in range(i + 1, min(i + 4, len(lines))):
        if not is_heading(lines[j]):
            next_items += 1

    # 🔥 must have at least 1–2 items after it
    return next_items >= 1


def is_bullet(line):
    return line.strip().startswith(("•", "-", "*"))

def clean_bullet(line):
    return line.lstrip("•-* ").strip()


# def extract_bullet_lists(text):
#     blocks = []
#     lines = [l.strip() for l in text.split("\n") if l.strip()]

#     current_title = None
#     current_items = []

#     for line in lines:

#         # 🔥 NEW HEADING
#         if is_heading(line):

#             # save previous block
#             if current_title and len(current_items) >= 1:
#                 blocks.append({
#                     "title": current_title.lower(),
#                     "items": current_items
#                 })

#             current_title = line
#             current_items = []

#         else:
#             # 🔥 IMPORTANT: STOP if this line looks like a heading word inside items
#             if any(h in line.lower() for h in [
#                 "facilities", "breakfast", "room service",
#                 "meeting room", "additional services"
#             ]):
#                 continue  # 🚨 prevent contamination

#             if current_title:
#                 current_items.append(line)

#     # last block
#     if current_title and len(current_items) >= 1:
#         blocks.append({
#             "title": current_title.lower(),
#             "items": current_items
#         })
#         print("🔍 CHECK HEADING:", line, "→", is_heading(line))

#     return blocks
def is_valid_item(line):
    line = line.strip()

    # too short → ignore
    if len(line) < 5:
        return False

    # looks like noise
    if line.lower() in ["yes", "no", "ok"]:
        return False

    return True


# def extract_bullet_lists(text):
#     blocks = []
#     lines = [l.strip() for l in text.split("\n") if l.strip()]

#     current_title = None
#     current_items = []

#     for i, line in enumerate(lines):

#         # 🔥 detect heading
#         if is_real_heading(lines, i):

#             # save previous block
#             if current_title and len(current_items) >= 1:
#                 blocks.append({
#                     "title": current_title.lower(),
#                     "items": current_items
#                 })

#             current_title = line
#             current_items = []

#         else:
#             if current_title and is_valid_item(line):

#                 # 🔥 avoid adding next heading accidentally
#                 if i + 1 < len(lines) and is_heading(lines[i + 1]):
#                     current_items.append(line)
#                 else:
#                     current_items.append(line)

#     # last block
#     if current_title and len(current_items) >= 1:
#         blocks.append({
# #             "title": current_title.lower(),
# #             "items": current_items
# #         })

# #     return blocks

# def extract_bullet_lists(text):
#     blocks = []
#     lines = [l.strip() for l in text.split("\n") if l.strip()]

#     current_title = None
#     current_items = []

#     for line in lines:

#         # 🔥 CASE 1: BULLET ITEM
#         if is_bullet(line):
#             if current_title:
#                 current_items.append(clean_bullet(line))
#             continue

#         # 🔥 CASE 2: NON-BULLET LINE → POSSIBLE HEADING
#         if is_heading(line):

#             # save previous block
#             if current_title and current_items:
#                 blocks.append({
#                     "title": current_title.lower(),
#                     "items": current_items
#                 })

#             current_title = line
#             current_items = []

#         # 🔥 CASE 3: fallback (non-bullet item)
#         else:
#             if current_title:
#                 current_items.append(line)

#     # last block
#     if current_title and current_items:
#         blocks.append({
#             "title": current_title.lower(),
#             "items": current_items
#         })

#     return blocks

def is_strong_heading(line):
    line = line.strip()

    # 🔥 1. Title Case (IMPORTANT)
    if line.istitle() and len(line.split()) <= 4:
        return True

    # 🔥 2. ALL CAPS
    if line.isupper():
        return True

    # 🔥 3. Known section keywords
    keywords = [
        "facilities", "services", "dishes",
        "menu", "beverages", "hours",
        "options", "amenities"
    ]

    if any(k in line.lower() for k in keywords):
        return True

    # ❌ reject sentences
    if any(w in line.lower() for w in [
        "available", "included", "free", "service"
    ]):
        return False

    return False



def extract_bullet_lists(text):
    blocks = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    current_title = None
    current_items = []

    for i, line in enumerate(lines):

        # -------------------------
        # 🔥 1. CONTACT BLOCK (HIGHEST PRIORITY)
        # -------------------------
        if is_contact_heading(line):

            # 🔥 save previous block
            if current_title and current_items:
                blocks.append({
                    "title": current_title.lower(),
                    "items": current_items
                })

            items = extract_contact_items(lines, i)
            parsed_items = parse_contact_items(items)

            if items:
                blocks.append({
                    "title": "contact information",
                    "items": items,
                    "structured": parsed_items,
                    "type": "contact"
                })

            # 🔥 RESET state (VERY IMPORTANT)
            current_title = None
            current_items = []

            continue  # 🚀 skip further processing

        # -------------------------
        # 🔥 2. NORMAL HEADING
        # -------------------------

        # -------------------------
        # 🔥 STRONG HEADING DETECTION
        # -------------------------
        if is_strong_heading(line):

            if current_title and current_items:
                blocks.append({
                    "title": current_title.lower(),
                    "items": current_items
                })

            current_title = line
            current_items = []

        else:
            # 🔥 ALWAYS treat as item if inside a block
            if current_title:
                current_items.append(line)

    # last block
    if current_title and current_items:
        blocks.append({
            "title": current_title.lower(),
            "items": current_items
        })

    return blocks



def protect_day_ranges(text):
    return re.sub(
        r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+to\s+'
        r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        lambda m: m.group(0).replace(" to ", "_TO_"),
        text,
        flags=re.I
    )


# ----------------------------------------
# MAIN PROCESS (UPDATED 🔥)
# ----------------------------------------
def process_documents(documents, business_id):

    fine_chunks = []     # 🔥 line-level (your current system)
    coarse_chunks = []   # 🔥 section-level (NEW)
    list_chunks = []   # ✅ NEW
    entity_section_map = {}

    for doc in documents:


        


            
        text = clean_text(doc["text"])

        print("\n" + "=" * 80)
        print("🔎 AFTER clean_text()")
        print("=" * 80)

        print(
            "SWIMMING POOL FOUND:",
            "swimming pool" in text.lower()
        )

        for line in text.splitlines():
            if "swimming" in line.lower() or "pool" in line.lower():
                print("MATCH:", repr(line))

        # 🔥 TWO VERSIONS
        text_for_time = text                     # original (safe)
        #text_for_list = restore_structure_for_lists(text)  # structured
        text_for_list = text

        text_for_time = protect_day_ranges(text_for_time)

        print("\n" + "=" * 80)
        print("🔎 AFTER protect_day_ranges()")
        print("=" * 80)

        print(
            "SWIMMING POOL FOUND:",
            "swimming pool" in text_for_time.lower()
        )

        for line in text_for_time.splitlines():
            if "swimming" in line.lower() or "pool" in line.lower():
                print("MATCH:", repr(line))

        text_for_list = protect_day_ranges(text_for_list)

        print("\n" + "=" * 80)
        print("🔎 AFTER protect_day_ranges() FOR LIST")
        print("=" * 80)

        print(
            "SWIMMING POOL FOUND:",
            "swimming pool" in text_for_list.lower()
        )

        for line in text_for_list.splitlines():
            if "swimming" in line.lower() or "pool" in line.lower():
                print("MATCH:", repr(line))

        print("\n🔍 RAW TEXT FOR LIST:\n", text_for_list[:500])
        source = doc["source"]

        # -----------------------------
        # 🔥 LIST BLOCK EXTRACTION (NEW)
        # -----------------------------
        list_blocks = extract_bullet_lists(text_for_list)

        for block in list_blocks:

            section = block["title"].strip().lower()

            for item in block.get("items", []):

                entity = resolve_entity_from_text(
                    item,
                    DOMAIN_ENTITY_REGISTRY
                )

                if entity:

                    entity_section_map[entity] = section

                    print(
                        "🧠 ENTITY → DOCUMENT SECTION:",
                        repr(item),
                        "→",
                        entity,
                        "→",
                        section
                    )

        print("\n========== ENTITY → SECTION MAP ==========")

        for entity, section in entity_section_map.items():
            print(
                repr(entity),
                "→",
                repr(section)
            )


        item_section_map = {}

        for block in list_blocks:

            section = block["title"].strip().lower()

            for item in block.get("items", []):

                normalized_item = item.strip().lower()

                item_section_map[normalized_item] = section


        print("\n========== ITEM → SECTION MAP ==========")

        for item, section in item_section_map.items():
            if (
                "laundry" in item
                or "swimming" in item
                or "room service" in item
            ):
                print(repr(item), "→", section)



        for block in list_blocks:

            section_name = block["title"]

            print(
                "SECTION:",
                section_name
            )

            for item in block["items"]:

                if (
                    "laundry" in item.lower()
                    or "swimming" in item.lower()
                    or "room service" in item.lower()
                ):
                    print(
                        "MATCH:",
                        item,
                        "→",
                        section_name
                    )

        print("\n" + "=" * 80)
        print("🔎 LIST EXTRACTION")
        print("=" * 80)

        for block in list_blocks:

            print("TITLE:", block["title"])

            for item in block["items"]:

                if (
                    "swimming" in item.lower()
                    or "pool" in item.lower()
                ):
                    print("🏊 SWIMMING POOL LIST ITEM:", repr(item))


        print("\n🧩 LIST BLOCKS DETECTED:", list_blocks)
        print("\n🧩 LIST BLOCKS DETECTED:")
        for block in list_blocks:
            print("TITLE:", block["title"])
            print("ITEMS:", block["items"])

        for block in list_blocks:
            list_chunks.append({
                "text": " | ".join(block["items"]),
                "type": block.get("type", "list"),  # 🔥 IMPORTANT
                "list_title": block["title"],
                "items": block["items"],
                "structured": block.get("structured", []),  # 🔥 ADD THIS
                "business_id": business_id,
                "source": source,
                "section": detect_section(block["title"])
            })

            print("\n✅ FINAL LIST CHUNKS:")
            for c in list_chunks:
                print(c["list_title"], "->", c["items"][:3])





        # 🔥 STEP 1: Try section splitting
        text = protect_time(text)
        text = fix_time_fragments(text)

        
        # -----------------------------
        # COARSE CHUNKS (LIST)
        # -----------------------------

        print("\n" + "=" * 80)
        print("🔎 INPUT TO split_into_sections()")
        print("=" * 80)
        
        for line in text_for_list.splitlines():
        
            if "swimming" in line.lower() or "pool" in line.lower():
                print("🏊 SECTION INPUT:", repr(line))

        sections = split_into_sections(text_for_list)

        # print("\n" + "=" * 80)
        # print("🔎 OUTPUT OF split_into_sections()")
        # print("=" * 80)

        # print("NUMBER OF SECTIONS:", len(sections))

        # for i, sec in enumerate(sections):

        #     print(f"\n--- SECTION {i} ---")
        #     print(repr(sec[:300]))

        #     if "swimming pool" in sec.lower():
        #         print("🏊🏊🏊 SWIMMING POOL SURVIVED SECTION SPLIT")


        print("\n========== SECTION → FINE CONTEXT DEBUG ==========")

        for section_text in sections:

            section_name = detect_section(section_text)

            print("\nSECTION:", repr(section_name))
            print("TEXT:", repr(section_text[:200]))

        

        if not sections or len(sections) <= 1:
            sections = split_into_sentences(text_for_list)

            # print("\n" + "=" * 80)
            # print("🔎 OUTPUT OF split_into_sentences()")
            # print("=" * 80)

            # for i, sec in enumerate(time_sections):

            #     if (
            #         "swimming" in sec.lower()
            #         or "pool" in sec.lower()
            #     ):
            #         print("🏊 FOUND IN SENTENCE:", i)
            #         print(repr(sec))


        print("☠️☠️☠️☠️☠️☠️☠️☠️")
        print(
            resolve_entity_from_text(
                    "Indoor swimming pool",
                    DOMAIN_ENTITY_REGISTRY
                )
            )
        
        print(
                resolve_entity_from_text(
                    "Laundry and dry cleaning",
                    DOMAIN_ENTITY_REGISTRY
                )
            )
        
        print(
                resolve_entity_from_text(
                    "Room service available from 6:00 AM - 11:00 PM",
                    DOMAIN_ENTITY_REGISTRY
                )
            )

        for sec in sections:

            #print("RAW SEC:", sec[:100])
            

            clean_sec = sec.replace("__TIME__", "").replace("_to_", " to ")
            section_name = detect_section(clean_sec)


            # -----------------------------
            # COARSE CHUNK
            # -----------------------------
            

            coarse_chunks.append({
                "text": clean_sec,
                "source": source,
                "business_id": business_id,
                "section": section_name
            })

            


            # # -----------------------------
            # # FINE CHUNK
            # # -----------------------------

            # print("\n" + "=" * 80)
            # print("🔎 BEFORE split_into_sentences()")
            # print("=" * 80)

            # print(
            #     "SWIMMING POOL FOUND:",
            #     "swimming pool" in text_for_time.lower()
            # )

        # REMOVE DUPLICATES
        seen = set()
        unique_coarse = []
        
        for c in coarse_chunks:
            if c["text"] not in seen:
                unique_coarse.append(c)
                seen.add(c["text"])

            # key = (
            #     chunk["text"],
            #     chunk["business_id"],
            #     chunk["section"]
            # )

            # if key in seen:
            #     continue

            # seen.add(key)
            # unique_coarse.append(chunk)
        
        coarse_chunks = unique_coarse


        time_sections = split_into_sentences(text_for_time)
        
        # print("\n" + "=" * 80)
        # print("🔎 AFTER split_into_sentences()")
        # print("=" * 80)
        
        # print(
        #     "SWIMMING POOL FOUND:",
        #     any(
        #         "swimming pool" in sec.lower()
        #         for sec in time_sections
        #         )
        #     )
        
        # for sec in time_sections:
        
        #     if "swimming" in sec.lower() or "pool" in sec.lower():
        
        #         print("🏊 SENTENCE CONTAINING POOL:")
        #         print(repr(sec))
        
        for sentence in time_sections:
        
            lines = clean_chunk_text(sentence)
                        
        
            # for line in lines:
        
            #     if (
            #         "swimming" in line.lower()
            #         or "pool" in line.lower()
            #         ):
            #         print("🏊 AFTER clean_chunk_text():", repr(line))
        
            for i, line in enumerate(lines):
        
                line = line.replace("__TIME__", "").replace("_to_", " to ")
                line = line.strip()
        
                if len(line) < 3:
                    continue
        
                # 🔥 MERGE "Sunday" + "Closed"
                DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
                if re.match(rf'^({"|".join(DAYS)})$', line.lower()):
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if "closed" in next_line.lower():
                            line = f"{line}: Closed"
        
                # 🔥 CAPTURE CLOSED (NOW WILL WORK)
                if re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday).*closed', line, re.I):
        
                    fine_chunks.append({
                        "text": line,
                        "source": source,
                        "business_id": business_id,
                        "section": "hours",
                        "type": "time"
                        })
        
                    continue  # 🔥 avoid duplicate append
        
                # -------------------------
                # NORMAL FLOW
                # -------------------------
                if len(line) < 8:
                    continue

                normalized_line = line.strip().lower()

                section = item_section_map.get(
                    normalized_line,
                    detect_section(line)
                )
        
                # fine_chunks.append({
                #     "text": line,
                #     "source": source,
                #     "business_id": business_id,
                #     "section": section
                #     })

                chunk = {
                    "text": line,
                    "source": source,
                    "business_id": business_id,
                    "section": section
                }

                print("\n" + "=" * 70)
                print("🧩 CREATING FINE CHUNK")
                print("=" * 70)
                print("INPUT LINE :", repr(line))
                print("CHUNK TEXT :", repr(chunk["text"]))
                print("SECTION    :", repr(chunk["section"]))

                fine_chunks.append(chunk)

                if (
                    "laundry" in line.lower()
                    or "swimming pool" in line.lower()
                    or "room service" in line.lower()
                ):
                    print(
                        "🎯 FINE CHUNK SECTION:",
                        repr(line),
                        "→",
                        repr(section)
                    )
        
                            
        
        
                if "to" in line.lower():
                    print("🔥 RANGE CHUNK:", line)

    expected = {
        "Indoor swimming pool",
        "Laundry and dry cleaning",
        "Room service available from 6:00 AM - 11:00 PM"
    }

    actual = {
        c["text"]
        for c in fine_chunks
    }

    print("\n" + "=" * 70)
    print("🔬 SENTENCE → FINE CHUNK CONTRACT TEST")
    print("=" * 70)

    for item in expected:
        print(
            repr(item),
            "→",
            item in actual
        )
        
    print("\n===== SEARCHING FOR LAUNDRY IN FINE CHUNKS =====")
            
    found = False
            
    for chunk in fine_chunks:
        if "laundry" in chunk["text"].lower():
            found = True
            print(chunk)
            
            print("FOUND:", found)    
            print(len(fine_chunks))
    count = 0
        
    for chunk in fine_chunks:
        if "laundry" in chunk["text"].lower():
            count += 1
        
            print("Laundry Chunks:", count)
        
    count = 0
        
    for chunk in fine_chunks:
        if "room service" in chunk["text"].lower():
            count += 1
        
            print("Room Service chunks:", count)    
        
                   
    print("🔥 TOTAL CHUNKS CREATED:", len(coarse_chunks))
    print("\n🚨 FINAL LIST_CHUNKS BEFORE RETURN:", len(list_chunks))


    print("\n========== FINE CHUNK SECTION DEBUG ==========")

    for chunk in fine_chunks:
        text = chunk["text"].lower()

        if (
            "laundry" in text
            or "swimming pool" in text
            or "room service" in text
        ):
            print(
                "TEXT:",
                repr(chunk["text"])
            )
            print(
                "SECTION:",
                repr(chunk.get("section"))
            )

    
    print("\n========== FINAL FINE CHUNK METADATA ==========")

    for c in fine_chunks:

        if (
            "laundry" in c["text"].lower()
            or "swimming pool" in c["text"].lower()
            or "room service" in c["text"].lower()
        ):
            print({
                "text": c["text"],
                "section": c["section"]
            })    
            
    return fine_chunks, coarse_chunks, list_chunks, entity_section_map


