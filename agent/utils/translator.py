from deep_translator import GoogleTranslator

# -----------------------------
# LANGUAGE DETECTION
# # -----------------------------
# def is_hindi(text):
#     return any('\u0900' <= c <= '\u097F' for c in text)
def is_hindi(text):
    if not text:
        return False

    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            return True
    return False


# -----------------------------
# TRANSLATE TO ENGLISH
# -----------------------------
def to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text


# -----------------------------
# TRANSLATE TO HINDI
# -----------------------------
def to_hindi(text):
    try:
        return GoogleTranslator(source='en', target='hi').translate(text)
    except:
        return text