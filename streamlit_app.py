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

# Field for the user's prompt
prompt = st.text_input("What would you like to learn about today?")

# Function to generate a response from GPT-2
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Generate and display the response
if st.button("Enter"):
    prompt.strip():
    response = generate_response(prompt)
    st.text_area("Response:", value=response, height=200)
    

# Title of the app
st.title("My Super Awesome GPT-2 Deployment!")
