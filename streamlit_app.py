import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cache the model and tokenizer loading to optimize performance
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

def main():
    st.title("Hugging Face GPT-2 Text Generator")

    # Create a text input field for the user's prompt
    user_prompt = st.text_input("Enter your prompt:")

    # Slider to control max length of the generated text
    max_length = st.slider("Max length of the response:", min_value=50, max_value=1024, value=100, step=50)

    # Temperature for randomness in generation
    temperature = st.slider("Temperature (creativity):", min_value=0.1, max_value=1.5, value=1.0, step=0.1)

    # Slider to control top-k (sampling parameter)
    top_k = st.slider("Top-k (limit token choices):", min_value=0, max_value=100, value=50, step=10)

    # Slider to control top-p (nucleus sampling)
    top_p = st.slider("Top-p (nucleus sampling):", min_value=0.0, max_value=1.0, value=0.95, step=0.05)

    # Generate a response based on the user's prompt
    if st.button("Generate"):
        if user_prompt.strip() == "":
            st.warning("Please enter a prompt before generating.")
        else:
            response = generate_response(user_prompt, max_length, temperature, top_k, top_p)
            st.text_area("Response:", value=response, height=200)

def generate_response(prompt, max_length, temperature, top_k, top_p):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs, 
            max_length=max_length, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

if __name__ == "__main__":
    main()
