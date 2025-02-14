# pip3 install langchain_openai
# python3 deepseek_langchain.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models.base import BaseChatOpenAI

llm = BaseChatOpenAI(
    model='deepseek-chat', 
    openai_api_key='OPENAI_API_KEY', 
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)

response = llm.invoke("Hi!")
print(response.content)