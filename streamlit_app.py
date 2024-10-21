import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
import os

### Load your API Key
my_secret_key = st.secrets['MyOpenAIKey']
os.environ["OPENAI_API_KEY"] = my_secret_key

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Title of the app
st.title("My Super Awesome GPT-2 Deployment!")

# Instructions
st.caption("Enter _what is on your mind_ followed by an auxilliary verb such as _is, was, has,_ etc. and see what happens! :sunglasses:")
st.caption("An example "Swimming is" has been provided below)


# Field for the user's prompt
prompt = st.text_input("What is on your mind?", "The weather is")

# Function to generate text
def generate_text(prompt, temperature=0.9, max_length=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_length=max_length,
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return gen_text

# Generate and display the response when the button is clicked
if st.button("Generate"):
    if prompt.strip():
        generated_text = generate_text(prompt)
        st.text_area("Generated Response:", value=generated_text, height=200)
    else:
        st.warning("Please enter a prompt!")
