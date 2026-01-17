# 📚 LLM-RAG-Document-QA

A **Retrieval-Augmented Generation (RAG)** system built from scratch that answers user questions strictly using provided documents, with **grounding, source attribution, and evaluation metrics** to reduce hallucinations.

---

## 🚀 Features

- 🔍 Semantic document retrieval using dense embeddings
- 🧠 LLM-based answer generation (Flan-T5)
- 📌 Source attribution for transparency
- 🧪 Evaluation metrics for grounding & faithfulness
- ⚙️ Modular, extensible architecture
- 🏗️ Built without LangChain (core concepts implemented manually)

---

## 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) combines:
- **Information Retrieval** → fetch relevant knowledge
- **Language Models** → reason and generate answers

This ensures answers are **grounded in documents**, not hallucinated.

---

## 🏗️ System Architecture

User Question
│
▼
Query Embedding
│
▼
Vector Similarity Search
│
▼
Top-K Relevant Chunks
│
▼
Context + Question Prompt
│
▼
LLM (Flan-T5)
│
▼
Answer + Sources + Evaluation


---

## 🧩 Pipeline Breakdown

### 1️⃣ Document Ingestion
- Loads text files from `data/documents/`
- Preserves source metadata (filename)

### 2️⃣ Chunking
- Documents are split into overlapping chunks
- Each chunk retains its source

### 3️⃣ Embedding
- Uses `all-MiniLM-L6-v2`
- Chunks and queries are embedded into the same vector space

### 4️⃣ Retrieval
- Cosine similarity used to rank chunks
- Top-K chunks retrieved
- Similarity threshold applied to avoid weak matches

### 5️⃣ Prompt Construction
- Retrieved chunks are passed as context
- Original user question is included to guide reasoning

### 6️⃣ Generation
- LLM generates answer **strictly from context**
- If context is insufficient → abstains

### 7️⃣ Source Attribution
- Displays which document chunks were used
- Improves trust and explainability

### 8️⃣ Evaluation
- **Recall@K** → checks retrieval quality
- **Context Coverage** → measures grounding
- **Faithfulness Check** → detects hallucination risk

---

## 📊 Evaluation Metrics

| Metric | Description |
|------|------------|
| Recall@K | Did we retrieve relevant chunks? |
| Context Coverage | How much answer overlaps with context |
| Faithfulness | Binary grounding decision |

---

## 📁 Project Structure
LLM-RAG-Document-QA/
│
├── app.py
│
├── data/
│ └── documents/
│
├── ingestion/
│ ├── load_documents.py
│ └── chunking.py
│
├── embeddings/
│ ├── embedding_model.py
│ └── generate_embeddings.py
│
├── retrieval/
│ └── similarity.py
│
├── llm/
│ ├── llm_model.py
│ ├── prompt.py
│ └── inference.py
│
├── evaluation/
│ ├── retrieval_metrics.py
│ ├── context_coverage.py
│ └── faithfulness.py
│
└── README.md

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python app.py

## Add your documents inside:

data/documents/

🧠 Key Learnings

Why retrieval alone is insufficient

How prompt + context work together

Importance of similarity thresholds

How to detect hallucinations

Real-world RAG evaluation strategies

🔮 Future Improvements

PDF ingestion

Streamlit UI

Vector database (FAISS)

Hugging Face deployment

Conversational memory

👤 Author

Anshu Pandey
Machine Learning & Deep Learning Practitioner
Focused on building systems from first principles
