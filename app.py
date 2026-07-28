import os
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

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 Think School AI Research Assistant")

st.caption(
    "AI-powered Research Assistant for Business Reports, Annual Reports & Case Studies"
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf"
    )

    if uploaded_file is not None:

        os.makedirs("uploaded_docs", exist_ok=True)

        pdf_path = os.path.join(
            "uploaded_docs",
            uploaded_file.name
        )

        # Process only once
        if st.session_state.processed_file != uploaded_file.name:

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Creating Knowledge Base..."):

                st.session_state.vector_db = create_vector_db(
                    pdf_path
                )

            st.session_state.processed_file = uploaded_file.name

            st.success("PDF Processed Successfully!")

            st.caption(f"📄 Document : {uploaded_file.name}")

            st.info("""
            💡 Suggested Questions

            • Summarize this company

            • Explain the business model

            • What are the biggest risks?

            • What are the revenue streams?
            """)

    st.divider()

    st.subheader("🧠 Think School Research Tools")

    selected_prompt = None

    for tool in PROMPTS.keys():

        if st.button(tool, use_container_width=True):

            selected_prompt = PROMPTS[tool]

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

    if user_question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

    if st.session_state.vector_db is None:

        st.warning("Please upload a PDF first.")

    else:

        with st.spinner("Thinking..."):

            answer, sources = ask_question(
                question,
                st.session_state.vector_db
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

            if sources:

                pages = []

                for doc in sources:

                    page = doc.metadata.get("page")

                    if page is not None:

                        pages.append(page + 1)

                if pages:

                    pages = sorted(set(pages))

                    st.divider()

                    st.caption("📖 Source Pages")

                    st.write(", ".join([f"Page {p}" for p in pages]))

st.divider()

st.caption(
    "Built by Sushant Shankar | Powered by OpenAI • LangChain • FAISS"
)