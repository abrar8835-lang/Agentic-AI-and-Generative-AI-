from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import load_prompt
load_dotenv()

model= ChatGroq(model="llama-3.1-8b-instant")

st.header("Research Tool")

#asking the user (Dynamic prompt)input

paper_input = st.selectbox("Select Research Paper",
                           ["Attention Is All You Need ", "BERT: Pre-training of deep bidirectional Transformers",
                            "GPT-3: Language Models are Few-Shot Learners",
                            "Diffusion Models Beat GANs on Image synthesis"])
style_input = st.selectbox("select Explanation Style",
                           ["Beginner-Friendly", "Technical", "code_Oriented", "Mathematical"])
length_input = st.selectbox("Select Explanation length",
                            ["Short (1-2Paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# calling the template created in json format
template=load_prompt('template.json')


if st.button("Summarize"):
    # creating chain to invoke both the template and model using prompt template instead of f strings
    chain = template | model
    result=chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)