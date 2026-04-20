import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1,
)
#---------------------------------------
# Store chat history
#---------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
#---------------------------------------
# Display history
#---------------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)
#----------------------------------------
# Input
#----------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    user_message = HumanMessage(content=user_input)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(user_input)

    response = llm.invoke(st.session_state.messages)

    ai_message = AIMessage(content=response.content)
    st.session_state.messages.append(ai_message)

    with st.chat_message("assistant"):
        st.write(response.content)