import streamlit as st
from .rag_pipeline import answer_query

st.set_page_config(page_title="Health RAG Chatbot", page_icon="💬", layout="centered")

# Custom CSS for a modern dark (black/charcoal) chat UI
st.markdown(
    """
    <style>
    body, .stApp { background-color: #18181b; }
    .stTextInput > div > div > input {
        background-color: #23232a;
        color: #f3f3f3;
        border: 1px solid #333;
    }
    .stButton > button {
        background-color: #35363a;
        color: #fff;
        border-radius: 4px;
        border: none;
        padding: 0.5em 1.5em;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #444;
    }
    .stMarkdown, .stTextInput, .stButton, .stTitle, .stInfo {
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #f3f3f3;
    }
    .chat-bubble {
        background: #23232a;
        color: #f3f3f3;
        border-radius: 12px;
        padding: 1em;
        margin-bottom: 1em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .sources {
        color: #b3b3b3;
        font-size: 0.95em;
        margin-top: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Health Information Assistant")
st.write("Ask a health question. Answers are strictly grounded in trusted sources and always cited.")

query = st.text_input("Your question:", "", key="input")

if st.button("Get Answer") and query.strip():
    with st.spinner("Retrieving answer..."):
        answer, chunks = answer_query(query)
    st.markdown('<div class="chat-bubble">', unsafe_allow_html=True)
    st.markdown(f"**Answer:**\n{answer}")
    st.markdown('</div>', unsafe_allow_html=True)
    if "I don’t know based on the provided corpus" in answer or "I don't know based on the provided corpus" in answer:
        st.info("This question could not be answered from the provided health documents.")
    st.markdown('<div class="sources">**Top Retrieved Sources:**<ul>', unsafe_allow_html=True)
    for c in chunks:
        st.markdown(f"<li>{c['doc_title']} (<a href='{c['doc_url']}' style='color:#b3b3b3;' target='_blank'>{c['doc_url']}</a>)</li>", unsafe_allow_html=True)
    st.markdown('</ul></div>', unsafe_allow_html=True) 