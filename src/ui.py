import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Enterprise Knowledge Intelligence",
    page_icon="📚",
    layout="wide",
)


st.title("Enterprise Knowledge Intelligence")
st.caption("Grounded answers from enterprise documents")


question = st.text_area(
    "Ask a question",
    placeholder="Example: Can LFs help students via MS Teams?",
    height=100,
)


top_k = st.slider(
    "Retrieval candidates",
    min_value=1,
    max_value=5,
    value=3,
)


if st.button("Ask", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching enterprise knowledge..."):

        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "top_k": top_k,
                },
                timeout=120,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            st.error(
                "Unable to connect to the RAG API. "
                "Make sure FastAPI is running."
            )
            st.caption(str(exc))
            st.stop()

    data = response.json()

    st.subheader("Answer")

    st.write(data["answer"])

    if data["sources"]:
        st.subheader("Sources")

        for source in data["sources"]:
            st.markdown(
                f"- **Page {source['page']}** "
                f"· similarity `{source['score']:.4f}` "
                f"· `{source['chunk_id']}`"
            )
    else:
        st.info(
            "No sufficiently relevant evidence was found "
            "in the indexed documents."
        )