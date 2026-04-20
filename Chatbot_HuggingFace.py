from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
llm=HuggingFaceEndpoint(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)
result = BaseModel.invoke([
    HumanMessage(content ="Who is the president of Iran?", max_length=100)]
)
print(result.content)