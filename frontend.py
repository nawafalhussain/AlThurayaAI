import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://localhost:8000/chat"

st.title("Help Desk Chatbot 💬")
st.markdown("Ask your HR or Technical questions below:")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input field (special for chatbot!)
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send to FastAPI backend
    response = requests.post(API_URL, json={"user_id": "user_123", "message": user_input})

    if response.status_code == 200:
        bot_reply = response.json()["response"]
        st.session_state.messages.append({"role": "bot", "content": bot_reply})
    else:
        st.session_state.messages.append({"role": "bot", "content": "❌ Error: Could not connect to server."})

    # Re-display updated chat
    st.rerun()

