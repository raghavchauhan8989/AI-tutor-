import streamlit as st
import requests

st.set_page_config(page_title="AI Tutor", page_icon="🎓")
st.title("🎓 AI Tutor")
st.caption("Ask me anything from the course material!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "student1"

mode = st.sidebar.radio("Mode", ["💬 Chat", "📝 Quiz"])

if mode == "💬 Chat":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = requests.post("http://127.0.0.1:8000/ask",
                    json={"question": prompt, "session_id": st.session_state.session_id})
                answer = res.json()["answer"]
                st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    topic = st.text_input("Enter a topic for quiz:")
    if st.button("Generate Quiz"):
        with st.spinner("Generating questions..."):
            res = requests.post("http://127.0.0.1:8000/quiz",
                json={"topic": topic, "session_id": st.session_state.session_id})
            questions = res.json()["questions"]
            for i, q in enumerate(questions):
                st.subheader(f"Q{i+1}: {q['question']}")
                for opt in q["options"]:
                    st.write(opt)
                st.success(f"Answer: {q['answer']}")
                st.divider()