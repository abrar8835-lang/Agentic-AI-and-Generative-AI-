from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()
model= ChatGroq(model="llama-3.1-8b-instant")

chat_history=[
    SystemMessage(content= 'you are a helpful AI assistant'),
]
while True:
    user_input = input('YOU:  ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result= model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ',result.content)
print (chat_history)