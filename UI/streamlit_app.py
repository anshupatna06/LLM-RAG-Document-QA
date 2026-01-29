import requests
import streamlit as st

# ==============================
# Config
# ==============================
st.set_page_config(page_title="RAG Document QA", layout="wide")
st.title("📚 LLM RAG Document Question Answering")

API_URL = "http://localhost:8000"

# ==============================
# Sidebar — Debug Controls
# ==============================
st.sidebar.header("🛠️ Debug Controls")

DEBUG_MODE = st.sidebar.checkbox("Enable Debug Mode", value=False)

TOP_K = st.sidebar.slider("Top-K Retrieved Chunks", 1, 10, 3)
SIMILARITY_THRESHOLD = st.sidebar.slider(
    "Similarity Threshold", 0.0, 1.0, 0.3, 0.05
)

# ==============================
# Sidebar — Upload Documents
# ==============================
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt or .pdf files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("Upload Documents"):
    for file in uploaded_files:
        files = {"file": (file.name, file.getvalue())}
        requests.post(f"{API_URL}/upload", files=files)
    st.sidebar.success("Documents uploaded & indexed successfully!")

# ==============================
# Sidebar — Delete Documents
# ==============================
st.sidebar.subheader("🗑️ Manage Documents")

docs_resp = requests.get(f"{API_URL}/documents").json()
documents = docs_resp.get("documents", [])

if documents:
    file_to_delete = st.sidebar.selectbox("Select document", documents)
    if st.sidebar.button("Delete selected document"):
        requests.delete(
            f"{API_URL}/documents",
            json={"filename": file_to_delete}
        )
        st.sidebar.success(f"Deleted {file_to_delete}")
else:
    st.sidebar.write("No documents available.")

# ==============================
# User Input
# ==============================
question = st.text_input("Radhe Radhe! Ask me anything based on the documents:")

response = None

if question:
    payload = {
        "question": question,
        "top_k": TOP_K,
        "threshold": SIMILARITY_THRESHOLD,
        "debug": DEBUG_MODE
    }

    try:
        response = requests.post(
            f"{API_URL}/query",
            json=payload,
            timeout=30
        ).json()
    except Exception:
        st.error("❌ Backend API is not reachable")
        st.stop()

# ==============================
# SUCCESS PATH
# ==============================
if response and response.get("answer"):

    rewritten_query = response["query"]["rewritten"]
    retrieval = response.get("retrieval", {})
    chunks = retrieval.get("chunks", [])

    used_chunks = [c for c in chunks if c["used"]]
    ignored_chunks = [c for c in chunks if not c["used"]]

    st.subheader("✅ Answer")
    st.write(response["answer"])

    # --------------------------
    # Sources
    # --------------------------
    st.subheader("📌 Sources")
    for src in response["sources"]:
        st.write("-", src)

    # ==============================
    # 🧠 Explainability View
    # ==============================
    st.divider()
    st.subheader("🧠 Why this answer?")

    st.markdown("### ✅ Used Context")
    if not used_chunks:
        st.info("No chunks passed the similarity threshold.")
    else:
        for c in used_chunks:
            with st.expander(
                f"📄 {c['source']} | score={c['score']:.3f}"
            ):
                st.markdown(
                    f"""
                    **Rank:** {c['rank']}  
                    **Similarity Score:** `{c['score']:.3f}`
                    """
                )
                st.write(c["text"])

    st.markdown("### ⚠️ Retrieved but Ignored")
    if ignored_chunks:
        for c in ignored_chunks:
            with st.expander(
                f"❌ {c['source']} | score={c['score']:.3f}"
            ):
                st.markdown(
                    f"""
                    **Rank:** {c['rank']}  
                    **Similarity Score:** `{c['score']:.3f}`  
                    **Reason:** Below similarity threshold
                    """
                )
                st.write(c["text"])
    else:
        st.success("All retrieved chunks were used 🎯")

    st.markdown("### 📊 Retrieval Summary")
    st.write(
        f"""
        • **Retrieved:** {retrieval.get('retrieved_chunks', 0)}  
        • **Used:** {retrieval.get('used_chunks', 0)}  
        • **Threshold:** {retrieval.get('threshold', 0)}
        """
    )

    # ==============================
    # Metrics
    # ==============================
    st.divider()
    st.subheader("📊 Evaluation Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Recall@K", response["metrics"]["recall_at_k"])
    c2.metric("Context Coverage", f"{response['metrics']['context_coverage']:.2f}")
    c3.metric(
        "Faithful Answer",
        "Yes" if response["metrics"]["faithful"] else "No"
    )

    st.subheader("🧠 Hallucination Check")
    grounding = response["metrics"]["grounding_score"]

    st.write(f"Grounding Score: **{grounding:.3f}**")

    if grounding < 0.6:
        st.error("⚠️ High hallucination risk")
    elif grounding < 0.75:
        st.warning("⚠️ Partial grounding")
    else:
        st.success("✅ Well grounded")

# ==============================
# FAILURE PATH
# ==============================
elif response:

    st.warning("I cannot find the answer in the provided documents.")

    failure = response.get("failure")
    if failure:
        st.divider()
        st.subheader("❌ Why was this answer refused?")

        st.error(f"**Failure Type:** {failure['type']}")

        st.markdown(
            f"""
            **Reason:** {failure['reason']}  
            **Similarity Threshold:** {failure['threshold']}  
            **Highest Retrieved Score:** `{failure['max_score']:.3f}`
            """
        )

        st.markdown("### 🛠️ What you can try")
        st.write("• Lower the similarity threshold")
        st.write("• Increase Top-K retrieval")
        st.write("• Upload more relevant documents")

# ==============================
# Debug Views
# ==============================
if DEBUG_MODE and response and response.get("retrieval"):
    st.divider()
    st.subheader("🔍 Retrieval Debug View")

    for c in response["retrieval"]["chunks"]:
        st.markdown(
            f"**Rank {c['rank']} | Score:** `{c['score']:.3f}` | Used: `{c['used']}`"
        )
        st.markdown(f"**Source:** `{c['source']}`")
        st.write(c["text"])
        st.markdown("---")

    scores = [c["score"] for c in response["retrieval"]["chunks"]]
    st.subheader("📊 Similarity Distribution")
    st.bar_chart(scores)

    st.sidebar.subheader("✍️ Query Debug")
    st.sidebar.code(f"Original:\n{question}")
    st.sidebar.code(f"Rewritten:\n{response['query']['rewritten']}")

# ==============================
# Performance Metrics
# ==============================
st.divider()
st.subheader("⏱️ Performance")

if not response:
    st.info("Run a query to see performance metrics.")
else:
    performance = response.get("performance", {})

    latency = performance.get("latency", {})
    cost = performance.get("cost", {})

    total_sec = latency.get("total_sec", 0.0)
    retrieval_sec = latency.get("retrieval_sec", 0.0)
    llm_sec = latency.get("llm_sec", 0.0)

    st.metric("Total Latency (sec)", round(total_sec, 3))
    st.caption(
        f"Retrieval: {retrieval_sec}s | LLM: {llm_sec}s"
    )

    # Interpretation
    if total_sec > 3:
        st.error("🐢 Slow response")
    elif total_sec > 1.5:
        st.warning("⏳ Moderate latency")
    else:
        st.success("⚡ Fast response")

    st.subheader("💰 Cost (Estimated)")

    prompt_tokens = cost.get("prompt_tokens", 0)
    completion_tokens = cost.get("completion_tokens", 0)
    total_tokens = cost.get("total_tokens", 0)
    estimated_cost = cost.get("estimated_cost_usd", 0.0)

    st.write(f"Prompt Tokens: {prompt_tokens}")
    st.write(f"Completion Tokens: {completion_tokens}")
    st.write(f"Total Tokens: {total_tokens}")
    st.write(f"Estimated Cost: ${estimated_cost}")

    if estimated_cost > 0.01:
        st.error("💸 High cost query")
    elif estimated_cost > 0.003:
        st.warning("💰 Moderate cost")
    else:
        st.success("🟢 Cheap query")
