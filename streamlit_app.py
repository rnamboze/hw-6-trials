rimport streamlit as st
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
st.subheader("Enter _what NOUN is on your mind_ followed by an auxilliary verb such as _is, was, has,_ etc. and see what happens! :sunglasses:", divider=True)
st.caption("The example _Coffee is_ has been provided below")

# Field for the user's prompt
prompt = st.text_input("What is on your mind?", "Coffee is")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
)    
gen_text = tokenizer.batch_decode(gen_tokens)[0]
 

# Generate and display the response when the button is clicked
st.write(
    response.choices[0].message.content
)
