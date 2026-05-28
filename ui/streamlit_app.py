import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Stock Assistant")

st.title("📈 Stock Assistant")

question = st.text_input("Ask a question")


if st.button("Submit") and question:
    payload = {
        "question": question,
        "user_id": "demo-user",
        "session_id": str(uuid.uuid4())
    }

    with st.spinner("Thinking..."):
        response = requests.post(API_URL, json=payload)

    data = response.json()

    if "answer" in data:
        st.success(data["answer"])

    elif "message" in data:
        st.warning(data["message"])