import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load API Key from .env
load_dotenv()

# 2. Page Configuration
st.set_page_config(page_title="Gemma 4 Chatbot", page_icon="🤖")
st.title("🤖 Gemma 4 Chatbot")

# 3. Initialize Google GenAI Client
@st.cache_resource
def get_client():
    return genai.Client()

client = get_client()

# 4. Initialize Chat Memory in Streamlit Session State
# Streamlit reruns the script on every user interaction, so session_state
# is used to preserve chat history between browser refreshes.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Past Chat Messages in UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input Bar
if user_input := st.chat_input("Type your message here..."):
    
    # Display User Message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Save User Message to Streamlit session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Convert session history into the payload structure Google API expects
    api_contents = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        api_contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    # Generate and Stream Bot Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Call Gemma 4 streaming endpoint
            response_stream = client.models.generate_content_stream(
                model="gemma-4-26b-a4b-it",
                contents=api_contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful and friendly AI assistant."
                )
            )

            # Stream response chunk by chunk to screen
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error calling Gemma API: {e}")

    # Save Bot Response to Streamlit session state
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})