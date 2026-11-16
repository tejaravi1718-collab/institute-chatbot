import streamlit as st

from rag import answer, build_index

st.set_page_config(page_title="Sunrise Institute Assistant", page_icon="🎓")


@st.cache_resource(show_spinner="Loading the assistant...")
def load_engine():
    return build_index()


model, index, chunks = load_engine()

st.title("Sunrise Institute Assistant")
st.caption("Demo: answers come only from the institute's own brochure.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me about fees, batch timings, syllabus or admissions.",
        }
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Answered from"):
                for s in m["sources"]:
                    st.text(s)

question = st.chat_input("Type your question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, found = answer(question, model, index, chunks)
            except Exception:
                reply, found = (
                    "Sorry, something went wrong. Please try again in a moment.",
                    [],
                )
        st.markdown(reply)
        sources = [c["text"] for c in found]
        if sources:
            with st.expander("Answered from"):
                for s in sources:
                    st.text(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )