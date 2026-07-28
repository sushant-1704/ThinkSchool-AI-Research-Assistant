import os
import hashlib
import streamlit as st

from backend import create_vector_db, ask_question
from prompts import PROMPTS

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Think School AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 Think School AI Research Assistant")

st.caption(
    "Analyze Annual Reports, DRHPs and Business Documents using Retrieval-Augmented Generation (RAG)."
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        os.makedirs("uploaded_docs", exist_ok=True)

        file_bytes = uploaded_file.getvalue()

        current_hash = hashlib.md5(file_bytes).hexdigest()

        pdf_path = os.path.join(
            "uploaded_docs",
            f"{current_hash}.pdf"
        )

        if (
            st.session_state.vector_db is None
            or st.session_state.file_hash != current_hash
        ):

            with open(pdf_path, "wb") as f:
                f.write(file_bytes)

            with st.spinner("Creating Knowledge Base..."):

                st.session_state.vector_db = create_vector_db(
                    pdf_path
                )

            st.session_state.file_hash = current_hash

        st.success("✅ PDF Ready")

        st.caption(f"📄 {uploaded_file.name}")

        st.info(
            """
💡 Suggested Questions

• Summarize this company

• Explain the business model

• What are the biggest risks?

• What are the revenue streams?
"""
        )

    st.divider()

    st.subheader("🧠 Think School Research Tools")

    selected_prompt = None

    for title, prompt in PROMPTS.items():

        if st.button(title, use_container_width=True):

            selected_prompt = prompt

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

user_question = st.chat_input(
    "Ask anything about the uploaded document..."
)

question = None

if selected_prompt:

    question = selected_prompt

elif user_question:

    question = user_question

# --------------------------------------------------
# Generate Response
# --------------------------------------------------

if question:

    if st.session_state.vector_db is None:

        st.warning("⚠️ Please upload a PDF first.")

    else:

        if user_question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            with st.chat_message("user"):

                st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer, sources = ask_question(
                        question,
                        st.session_state.vector_db
                    )

                except Exception as e:

                    st.error(f"Error: {e}")
                    st.stop()

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            if sources:

                pages = sorted(
                    {
                        doc.metadata.get("page", 0) + 1
                        for doc in sources
                        if doc.metadata.get("page") is not None
                    }
                )

                if pages:

                    st.divider()

                    st.caption("📖 Source Pages")

                    st.write(
                        ", ".join(
                            [f"Page {page}" for page in pages]
                        )
                    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Built by Sushant Shankar | Powered by OpenAI • LangChain • FAISS"
)