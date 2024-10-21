from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def main():
    st.title("Hugging Face Text Generator")

    # Create a text input field for the user's prompt
    user_prompt = st.text_input("Enter your prompt:")

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

if __name__ == "__main__":
    main()
