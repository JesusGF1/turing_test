
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    #model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium") 
    #tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("AIDynamics/DialoGPT-medium-MentorDealerGuy")
    tokenizer = AutoTokenizer.from_pretrained("AIDynamics/DialoGPT-medium-MentorDealerGuy")
    return model, tokenizer

def generate_response(prompt, chat_history_ids, model, tokenizer):
    # encode prompt
    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # append to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if len(chat_history_ids) > 0 else new_user_input_ids
    
    # generate response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # decode response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

st.title('🎈 Turing test chatbot')
st.write('In this app you will be able to chat with a chatbot that has been trained on a dataset of conversations between two people. The chatbot is based on Large Lenguage models, in particular GPT2')
st.write('However, if there are some of you running the app at the same time there is a chance to chat with each other. ')
model, tokenizer = load_model() 
chat_history_ids = torch.zeros((1,1), dtype=torch.long)
i = 0 
while i < 20:
    user_input = st.text_input("You: ", "",key=str(i)) 
    if st.button('Send',key=str(i)+"a"):
        if user_input:
            output, chat_history_ids = generate_response(user_input, chat_history_ids, model, tokenizer)
            st.text_area("User:", output, height=100, key=f"{len(chat_history_ids)}b")
            st.empty()
    i += 1

if st.button('Who was I speaking with?',key="Speaker"):
    st.write("You were speaking with a chatbot")