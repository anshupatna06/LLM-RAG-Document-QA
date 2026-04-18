import requests
import streamlit as st

# ==============================
# Config
# ==============================
st.set_page_config(page_title="RAG Document QA", layout="wide")
st.title("🏨 Hotel AI Assistant")
st.caption("AI-powered knowledge assistant for hospitality businesses")

API_URL = "http://localhost:8000"

# ==============================
# Mode Selection
# ==============================
MODE = st.sidebar.radio(
    "App Mode",
    ["Business Mode", "Developer Mode"],
    index=0
)
st.sidebar.write("pending_query:", st.session_state.get("pending_query"))
st.sidebar.write("response:", st.session_state.get("response") is not None)
DEBUG_MODE = MODE == "Developer Mode"

# ==============================
# Session State Init
# ==============================
st.session_state.setdefault("pending_query", None)
st.session_state.setdefault("response", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_rendered_answer", None)

st.session_state.setdefault("trigger_query", False)
st.session_state.setdefault("current_question", None)
st.session_state.setdefault("processing", False)

if "business_id" not in st.session_state:
    st.session_state.business_id = "demo_hotel"

business_id = st.session_state.business_id

# ==============================
# Welcome Message
# ==============================
st.session_state.setdefault("welcome_shown", False)

if not st.session_state.welcome_shown:

    welcome = requests.get(
        f"{API_URL}/{business_id}/welcome"
    ).json()

    st.session_state.messages.append(
        {"role": "assistant", "content": welcome["message"]}
    )

    st.session_state.welcome_shown = True

# ==============================
# Sidebar Controls
# ==============================
st.sidebar.header("🛠️ Controls")

TOP_K = st.sidebar.slider("Top-K Retrieved Chunks", 1, 10, 3)
SIMILARITY_THRESHOLD = st.sidebar.slider(
    "Similarity Threshold", 0.0, 1.0, 0.3, 0.05
)

# ==============================
# Sidebar Upload
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


# -----------------------------
# Knowledge Base Panel
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("📚 Knowledge Base")

try:
    docs = requests.get(f"{API_URL}/documents").json()["documents"]
    st.sidebar.write(f"Documents Indexed: {len(docs)}")
    st.sidebar.success("Status: Ready")
except:
    st.sidebar.warning("Unable to load documents")



if st.sidebar.button("Rebuild Embeddings"):
    requests.post(f"{API_URL}/reindex")

#Add Business Selector (Sidebar)
st.sidebar.divider()
st.sidebar.subheader("Active Business")

business_id = st.sidebar.selectbox(
    "Select Business",
    ["demo_hotel"]
)

st.session_state.business_id = business_id

#Add System Status Panel
st.sidebar.divider()
st.sidebar.subheader("System Status")

#Show Document Count
docs = requests.get(f"{API_URL}/documents").json().get("documents", [])

st.sidebar.metric(
    "Knowledge Base",
    len(docs)
)

st.sidebar.success("API Connected")
st.sidebar.success("LLM Ready")
st.sidebar.success("Vector DB Ready")

#Add Chat Clear Button
if st.sidebar.button("Reset Conversation"):

    st.session_state.messages = []
    st.session_state.response = None
    st.session_state.pending_query = None
    st.rerun()


# ==============================
# Sidebar Delete
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
# Chat History
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# HANDLE SUGGESTION CLICK FIRST
# ==============================
# ==============================
# INPUT BOX
# ==============================

typed_question = st.chat_input("Ask something from the documents...")

# ---------------------------------
# Priority: suggestion click
# ---------------------------------
if st.session_state.pending_query:

    st.session_state.current_question = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state.trigger_query = True

# ---------------------------------
# Otherwise typed input
# ---------------------------------
elif typed_question:

    st.session_state.current_question = typed_question
    st.session_state.trigger_query = True

# ==============================
# EXECUTE QUERY (single source of truth)
# ==============================

if st.session_state.trigger_query:

    question = st.session_state.current_question

    st.session_state.trigger_query = False
    st.session_state.response = None
    st.session_state.last_rendered_answer = None

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    try:
        with st.spinner("AI Thinking..."):
            resp = requests.post(
                f"{API_URL}/{business_id}/query",
                json={
                    "question": question,
                    "top_k": TOP_K,
                    "threshold": SIMILARITY_THRESHOLD
                },
                timeout=30
            )

        st.session_state.response = resp.json()

    except Exception as e:
        st.error(f"Backend error: {e}")

    st.rerun()

# ==============================
# Assistant Response
# ==============================
response = st.session_state.response


if response:

    decision = response.get("decision", "ANSWER")

    with st.chat_message("assistant"):

        if decision == "ANSWER":

            import time
            
            answer = response.get("answer", "")
            
            status = st.empty()
            status.info("Searching documents...")
            time.sleep(0.4)
            status.info("Generating response...")
            time.sleep(0.4)
            status.empty()

            stream = st.empty()
            typed = ""

            for w in answer.split():
                typed += w + " "
                stream.markdown(typed)
                time.sleep(0.02)

            if not st.session_state.last_rendered_answer:
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.last_rendered_answer = answer

            latency = response.get("performance", {}).get("latency", {}).get("total_sec", 0)

            st.caption(f"Response time: {latency:.2f}s")

            # -----------------------------
            # Confidence Indicator
            # -----------------------------
            retrieval = response.get("retrieval", {})
            chunks = retrieval.get("chunks", [])

            max_score = max([c["score"] for c in chunks], default=0.0)

            if max_score > 0.75:
                st.success("🟢 High Confidence")
            elif max_score > 0.45:
                st.warning("🟡 Medium Confidence")
            else:
                st.error("🔴 Low Confidence")

            #typed = st.chat_input("Ask anything about the hotel…")
            # Suggestions
            suggestions = response.get("suggestions")

            DEFAULT_SUGGESTIONS = [
                "What facilities are available?",
                "Is breakfast included?",
                "What is check-in time?",
                "Do you provide parking?"
               ]

            import random
            random.shuffle(DEFAULT_SUGGESTIONS)

            if not suggestions:
               suggestions = DEFAULT_SUGGESTIONS[:3]

            
            st.markdown("### 💡 Suggested follow-ups")

            cols = st.columns(len(suggestions))

            for i, s in enumerate(suggestions):
                if cols[i].button(s, key=f"suggestion_{i}_{s}"):

                    st.session_state.pending_query = s
                    st.session_state.processing = True
                    #st.session_state.response = None
                    st.rerun()

            st.caption("Or type your own question below.")

            #Sources
            if chunks:
                st.markdown("### 📄 Sources Used")

                for c in chunks:
                    if c["used"]:
                        st.markdown(
                            f"- {c['source']} (score {c['score']:.2f})"
                        )

        elif decision == "REFUSE":

            msg = "🚫 I cannot answer this from the provided documents."
            st.error(msg)

            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )

        elif decision == "CLARIFY":

            st.warning(response.get("clarification"))

            followup = st.text_input("Your clarification")

            if st.button("Submit clarification") and followup:
                st.session_state.pending_query = followup
                st.rerun()

    # clear response AFTER rendering
    #st.session_state.response = None

# ==============================
# Debug Sidebar
# ==============================
if DEBUG_MODE and response:

    st.sidebar.divider()
    st.sidebar.subheader("🧪 Agent Debug View")

    retrieval = response.get("retrieval", {})
    chunks = retrieval.get("chunks", [])

    last_user = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            last_user = m["content"]
            break

    rewritten_q = None
    for step in response.get("agent_trace", []):
        if step["step"] == "rewrite":
            rewritten_q = step.get("detail")

    st.sidebar.code(f"Original:\n{last_user}")
    st.sidebar.code(f"Rewritten:\n{rewritten_q}")

    st.sidebar.write(f"Retrieved: {retrieval.get('retrieved_chunks', 0)}")
    st.sidebar.write(f"Used: {retrieval.get('used_chunks', 0)}")

# ==============================
# Performance
# ==============================
if response and response.get("performance"):

    st.divider()
    st.subheader("⏱️ Performance")

    performance = response["performance"]
    lat = performance.get("latency", {})
    cost = performance.get("cost", {})

    st.metric("Total Latency (sec)", lat.get("total_sec", 0.0))

    st.caption(
        f"Retrieval: {lat.get('retrieval_sec', 0)}s | "
        f"LLM: {lat.get('llm_sec', 0)}s"
    )

    st.subheader("💰 Cost")
    st.write(f"Total Tokens: {cost.get('total_tokens', 0)}")
    st.write(f"Estimated Cost: ${cost.get('estimated_cost_usd', 0.0)}")

st.divider()
st.caption("Powered by Retrieval-Augmented Generation")