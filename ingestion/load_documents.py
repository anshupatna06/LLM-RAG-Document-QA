import os

def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({
            "text": text,
            "source": filename
        })

    return documents
