import os
import pdfplumber
import re

def normalize_pdf_text(text):

    # normalize unicode dashes
    text = text.replace("–", "-").replace("—", "-")

    # fix spacing issues
    #text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[ \t]+", " ", text)   # keep newlines

    # 🔥 FIX: ensure AM/PM sticks to time
    text = re.sub(r'(\d{1,2}:\d{2})\s*(AM|PM)', r'\1 \2', text, flags=re.I)

    # 🔥 FIX: restore missing PM in ranges like "6:00 AM - 11:00"
    text = re.sub(
        r'(\d{1,2}:\d{2}\s*AM)\s*-\s*(\d{1,2}:\d{2})(?!\s*(AM|PM))',
        r'\1 - \2 PM',
        text,
        flags=re.I
    )

    return text.strip()


def fix_time_fragments(text):

    # merge broken time ranges split across lines
    text = re.sub(
        r'(\d{1,2}:\d{2}\s?(AM|PM))\s*\n\s*(\d{1,2}:\d{2}\s?(AM|PM))',
        r'\1 - \3',
        text,
        flags=re.I
    )

    # fix cases like "(7:00 AM\n10:00 AM)"
    text = re.sub(
        r'\(\s*(\d{1,2}:\d{2}\s?(AM|PM))\s*\n\s*(\d{1,2}:\d{2}\s?(AM|PM))\s*\)',
        r'(\1 - \3)',
        text,
        flags=re.I
    )
    

    return text



def load_documents(folder_path):

    documents = []

    for root, _, files in os.walk(folder_path):

        for filename in files:

            file_path = os.path.join(root, filename)

            # ---------- TXT ----------
            if filename.lower().endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()

                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": file_path   # ✅ FIXED
                        })

                except Exception as e:
                    print(f"Skipping TXT {filename}: {e}")

            # ---------- PDF ----------
            elif filename.lower().endswith(".pdf"):
                try:
                    text = ""

                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                #print("RAW:", page_text)
                                page_text = normalize_pdf_text(page_text)   # 🔥 ADD THIS
                                page_text = fix_time_fragments(page_text)   # 🔥 ADD THIS
                                text += page_text + "\n"
                                
                                #print("NORMALIZED:", normalize_pdf_text(page_text))

                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": file_path   # ✅ FIXED
                        })

                except Exception as e:
                    print(f"Skipping PDF {filename}: {e}")

            

    return documents