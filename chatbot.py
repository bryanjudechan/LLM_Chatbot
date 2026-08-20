import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Automatically load GEMINI_API_KEY from the .env file
load_dotenv()

# The client auto-detects GEMINI_API_KEY from environment variables
client = genai.Client()

# Maintain conversation state
chat_history = []
system_instruction = "You are Bryan's helpful and friendly assistant."

print("--- Gemma 4 Chatbot (Cloud API) ---")
print("Type 'exit' or 'quit' to end.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append user input to context
    chat_history.append({"role": "user", "parts": [{"text": user_input}]}) #"parts" needs to be a list of dicts even though we only have one part here

    try:
        # Call Gemma 4 via API
        response = client.models.generate_content(
            model="gemma-4-26b-a4b-it",
            contents=chat_history,
            config=types.GenerateContentConfig( # this line is optional, but allows you to set parameters like temperature, max output tokens, etc.
                system_instruction=system_instruction # refer to line 14 for the system instruction
            )
        )

        bot_reply = response.text
        print(f"\nBot: {bot_reply}\n")

        # Append model reply to history
        chat_history.append({"role": "model", "parts": [{"text": bot_reply}]})

    except Exception as e:
        print(f"\nError calling API: {e}\n")