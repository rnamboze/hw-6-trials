import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
import os

### Load your API Key
my_secret_key = st.secrets['MyOpenAIKey']
os.environ["OPENAI_API_KEY"] = my_secret_key

def main():
    st.title("GPT-2 Text Generator")

    # Create a text input field for the user's prompt
    user_prompt = st.text_input("Enter your prompt:")

    # Generate a response based on the user's prompt
    if st.button("Generate"):
        response = generate_response(user_prompt)
        st.text_area("Response:", value=response, height=200)

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="gpt2",  # Adjust the engine as needed
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7,  # Adjust temperature for creativity vs. coherence
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

if __name__ == "__main__":
    main()
