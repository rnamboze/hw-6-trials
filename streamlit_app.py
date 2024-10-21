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
st.subheader("Enter _what NOUN is on your mind_ followed by an auxilliary verb such as _is, was, has,_ etc. and see what happens! :sunglasses:", divider=True)
st.caption("The example _Coffee is_ has been provided below")

# Field for the user's prompt
prompt = st.text_input("What is on your mind?", "Coffee is")

# Field for the user to specify the number of tokens
num_tokens = st.number_input("Number of Tokens:", min_value=10, max_value=300, value=50)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Information on creativity and predictability obtained from https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/ - I have chosen to use temperature and Top-k and Top-p parameters

creative_response = model.generate(
        input_ids,
        do_sample=True,
        max_length=num_tokens,
        top_k=50,
        top_p=0.9,
        temperature=1.0,
)    
creative_text = tokenizer.batch_decode(creative_response)[0]

predictable_response = model.generate(
        input_ids,
        do_sample=True,
        max_length=num_tokens,
        top_k=50,
        top_p=0.9,
        temperature=0.2
)    
predictable_text = tokenizer.batch_decode(predictable_response)[0]

# Display the result
st.subheader("Creative Response:")
st.write(creative_text)

st.subheader("Predictable Response:")
st.write(predictable_text)

# The testing of this feature is through observation of the results displayed. 
# At a lower temperature, the model is more conservative and is limited to choosing tokens with higher probabilities. 
# At higher temperature, that limit gets lenient, allowing the model to choose lesser probable words, resulting in more unpredictable and creative text.
