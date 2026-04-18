import re


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
def split_into_sections(text):

    # split when a heading-like pattern appears
    sections = re.split(
        r'(?=\n?[A-Z][A-Za-z\s]{3,}\n)',
        text
    )

    return [s.strip() for s in sections if len(s.strip()) > 30]


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
# ----------------------------------------
def split_into_sentences(text):

    sentences = re.split(
        r'(?:\n+|•)',   # 🔥 REMOVE "-\s+"
        text
    )
    # sentences = re.split(
    #     r'(?:\n+|•|\.\s+(?=[A-Z]))',
    #     text
    # )

    return [s.strip() for s in sentences if len(s.strip()) > 10]


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



def clean_chunk_text(text):

    lines = re.split(r'(?:\n|•)', text)

    clean_lines = []

    for line in lines:

        line = line.strip()

        if len(line) < 5:
            continue

        # remove broken headings
        if line.endswith("("):
            continue

        if len(line.split()) <= 2:
            continue

        # ❌ remove broken headings / fragments
        if re.match(r'^[A-Za-z\s\-&()]+$', line) and len(line.split()) <= 3:
            continue

        # ❌ remove words like "Grill", "Options"
        if len(line.split()) == 1:
            continue

        merged_lines = []
        buffer = ""

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # 🔥 if previous line incomplete → merge
            if buffer and not buffer.endswith(('.', ':')):
                buffer += " " + line
            else:
                if buffer:
                    merged_lines.append(buffer)
                buffer = line

        if buffer:
            merged_lines.append(buffer)

        lines = merged_lines

        # if len(line.split()) < 3:
        #     continue

        # normalize dash
        line = line.replace("–", "-").replace("—", "-")

        clean_lines.append(line)

    return clean_lines    # return LIST instead of string Because we'll further use t = text.lower().replace("\n", " ") in build_time_map function


def is_heading(line):

    line = line.strip()

    # ❌ reject time lines
    if re.search(r'\d{1,2}:\d{2}', line):
        return False

    # ❌ reject very long lines
    if len(line.split()) > 5:
        return False

    # ✅ allow title-like phrases
    if line.istitle():   # <-- KEY FIX
        return True

    # ✅ fallback: contains keywords
    if any(k in line.lower() for k in [
        "dishes", "menu", "services", "facilities",
        "hours", "beverages", "options"
    ]):
        return True

    return False


def extract_bullet_lists(text):

    blocks = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    current_title = None
    current_items = []

    for line in lines:

        # 🔥 detect heading (short + no numbers)
        if is_heading(line) and not re.search(r'\d{1,2}:\d{2}', line):

            # save previous block
            if current_title and len(current_items) >= 2:
                blocks.append({
                    "title": current_title.lower(),
                    "items": current_items
                })

            current_title = line
            current_items = []

        else:
            current_items.append(line)

    # last block
    if current_title and len(current_items) >= 2:
        blocks.append({
            "title": current_title.lower(),
            "items": current_items
        })

        print("🔍 CHECK HEADING:", line, "→", is_heading(line))

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

    for doc in documents:

            
        text = clean_text(doc["text"])

        # 🔥 TWO VERSIONS
        text_for_time = text                     # original (safe)
        #text_for_list = restore_structure_for_lists(text)  # structured
        text_for_list = text

        text_for_time = protect_day_ranges(text_for_time)
        text_for_list = protect_day_ranges(text_for_list)

        print("\n🔍 RAW TEXT FOR LIST:\n", text_for_list[:500])
        source = doc["source"]

        # -----------------------------
        # 🔥 LIST BLOCK EXTRACTION (NEW)
        # -----------------------------
        list_blocks = extract_bullet_lists(text_for_list)
        print("\n🧩 LIST BLOCKS DETECTED:", list_blocks)
        print("\n🧩 LIST BLOCKS DETECTED:")
        for block in list_blocks:
            print("TITLE:", block["title"])
            print("ITEMS:", block["items"])

        for block in list_blocks:
            list_chunks.append({
                "text": " | ".join(block["items"]),
                "type": "list",
                "list_title": block["title"],
                "items": block["items"],
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
        sections = split_into_sections(text_for_list)

        if not sections or len(sections) <= 1:
            sections = split_into_sentences(text_for_list)

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
            # REMOVE DUPLICATES
            seen = set()
            unique_coarse = []

            for c in coarse_chunks:
                if c["text"] not in seen:
                    unique_coarse.append(c)
                    seen.add(c["text"])

            coarse_chunks = unique_coarse


            # # -----------------------------
            # # FINE CHUNK
            # # -----------------------------
            time_sections = split_into_sentences(text_for_time)

            for sec in time_sections:

                lines = clean_chunk_text(sec)

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

                    fine_chunks.append({
                        "text": line,
                        "source": source,
                        "business_id": business_id,
                        "section": detect_section(line)
                    })


                    if "to" in line.lower():
                        print("🔥 RANGE CHUNK:", line)
            

           
    print("🔥 TOTAL CHUNKS CREATED:", len(coarse_chunks))
    print("\n🚨 FINAL LIST_CHUNKS BEFORE RETURN:", len(list_chunks))

    return fine_chunks, coarse_chunks, list_chunks


# # print("SECTIONS:", len(sections))
#         # for s in sections[:5]:
#         #     #print("SEC:", s[:80])
#         #     #print("TEXT SAMPLE:", text[:300])

#         for sec in sections:

#             #print("RAW SEC:", sec[:100])
#             s = clean_chunk_text(sec)
#             s = s.replace("__TIME__", "")
#             s = s.replace("_to_", " to ")
#             #print("CLEAN SEC:", s[:100], "| LENGTH:", len(s))
#             print("SECTIONS:", len(sections))
