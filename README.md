📚 LLM-Powered RAG Document Question Answering System

A production-style Retrieval-Augmented Generation (RAG) system that answers questions from user-uploaded documents with explainability, evaluation metrics, failure analysis, latency & cost monitoring.

This project focuses not just on getting answers, but on understanding why an answer was generated or refused — a critical requirement for real-world LLM systems.

🚀 Key Highlights

✅ End-to-end RAG pipeline (Retrieval → Filtering → Generation)

🧠 Explainability dashboard (used vs ignored context)

📊 Evaluation metrics (recall@k, coverage, faithfulness, grounding)

❌ Failure-case analysis (why the model refused to answer)

⏱️ Latency breakdown (retrieval vs LLM)

💰 Cost estimation (token usage & estimated cost)

🧩 Modular, extensible architecture

🌐 Local + public demo support (ngrok)

🐳 Docker-ready (explored for cloud deployment)

🏗️ System Architecture
User Query
   ↓
Query Rewriting
   ↓
Vector Retrieval (Top-K)
   ↓
Similarity Filtering (Threshold)
   ↓
Context Selection
   ↓
LLM Answer Generation
   ↓
Evaluation + Explainability + Metrics

🧠 Core Concepts Implemented
🔹 Retrieval-Augmented Generation (RAG)

Prevents hallucination by grounding answers in retrieved document chunks

Uses similarity-based filtering to control relevance

🔹 Explainability (Why this answer?)

Shows:

Which chunks influenced the answer

Which chunks were retrieved but ignored

Why certain context was rejected

🔹 Failure-Case Dashboard

When no answer is generated, the system explains:

Similarity threshold violation

Highest retrieved score

Concrete steps to fix the issue (lower threshold, increase Top-K, add documents)

🔹 Evaluation Metrics

Recall@K – retrieval quality

Context Coverage – how much of the answer is grounded

Faithfulness – consistency with retrieved context

Grounding Score – hallucination risk indicator

🔹 Performance Monitoring

Retrieval latency

LLM latency

Total request latency

Token usage & estimated cost

🗂️ Project Structure
LLM-RAG-Document-QA/
│
├── app.py                  # FastAPI backend (API version)
├── streamlit_app.py        # Streamlit UI (direct pipeline version)
│
├── backend/
│   ├── rag_service.py      # Core RAG orchestration
│   ├── state.py            # Global state & embeddings
│   └── schemas.py          # Request / response schemas
│
├── rag_core/
│   └── pipeline.py         # RAG pipeline abstraction
│
├── retrieval/
│   └── similarity.py       # Vector similarity retrieval
│
├── llm/
│   ├── llm_model.py        # LLM wrapper
│   └── utils.py            # Token estimation
│
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── context_coverage.py
│   ├── faithfulness.py
│   └── hallucination.py
│
├── data/                   # Uploaded documents
├── requirements.txt
├── Dockerfile              # (Explored for deployment)
└── README.md

🖥️ Running Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit app
streamlit run streamlit_app.py

3️⃣ Open in browser
http://localhost:7860

🌍 Public Demo (Optional)

The app was successfully exposed using ngrok for mobile and external access:

ngrok http 7860


This generates a public HTTPS URL usable on any device.

🐳 Deployment Notes (Important)

Docker-based deployment was explored (Hugging Face Spaces)

Due to:

heavy initialization

embedding state

RAG pipeline startup costs

Hugging Face Spaces showed intermittent runtime issues

➡️ This is a platform limitation, not an architectural flaw.

In real-world setups, this system is better suited for:

AWS EC2 / ECS

Azure App Service

GCP Cloud Run

🎯 Why This Project Matters

This project goes beyond toy RAG demos by addressing real production concerns:

Explainability (trust)

Failure analysis (debuggability)

Cost awareness (scalability)

Performance monitoring (latency)

These are the exact concerns evaluated in:

ML engineer interviews

Applied AI roles

Startup MVP discussions

🔮 Future Extensions

Multimodal RAG (PDF + images)

Hybrid retrieval (BM25 + vectors)

Query intent classification

RAG evaluation automation

Agent-based document workflows

Cloud-native deployment (AWS/GCP)

👤 Author

Anshu Pandey
Aspiring Machine Learning & AI Engineer
Focused on building scalable, explainable ML systems
