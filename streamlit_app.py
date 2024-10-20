import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

st.title("GPT-2 Text Generation Web App")

# User input for the prompt
prompt = st.text_area("Enter your prompt:", "Once upon a time")

# User input for number of tokens
max_length = st.number_input("Number of tokens for the response:", min_value=10, max_value=200, value=50, step=10)

# Creativity options: high creativity vs. more predictable response
creativity = st.radio("Choose the creativity level:", ("Creative (High)", "Predictable (Low)"))

if st.button("Generate Response"):
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Set the parameters for generating responses
    if creativity == "Creative (High)":
        output = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,  # enables more random sampling
            top_k=50,        # limit to top 50 probable tokens to enhance creativity
            top_p=0.95,      # use nucleus sampling
            temperature=1.2  # higher temperature increases randomness
        )
    else:
        output = model.generate(
            inputs,
            max_length=max_length,
            do_sample=False,  # deterministic (greedy) sampling
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Display the result
    st.subheader("Generated Response:")
    st.write(generated_text)
