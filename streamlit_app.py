import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

def main():
    st.title("Hugging Face GPT-2 Text Generator")

    # Create a text input field for the user's prompt
    user_prompt = st.text_input("What would you like to learn about today:")

    # Generate a response based on the user's prompt
    if st.button("Enter"):
            response = generate_response(user_prompt)
            st.text_area("Response:", value=response, height=200)

def generate_response(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=1024, num_beams=4)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

if __name__ == "__main__":
    main()
