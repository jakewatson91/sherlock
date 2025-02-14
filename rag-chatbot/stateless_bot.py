import os
import warnings
from dotenv import load_dotenv
import pickle
import numpy as np
from huggingface_hub import InferenceClient
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

warnings.filterwarnings("ignore")

load_dotenv()
test_msg = "Give me a quick summary of Jake's experience"

# Convert document embedding to NumPy arra
with open('data/embedding_pairs.pkl', 'rb') as f:
    embedding_pairs = pickle.load(f)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                                openai_api_type=os.environ.get("OPENAI_API_KEY")
                                )
# Create and index the document
vectorstore = FAISS.from_embeddings(embedding_pairs, embeddings)

# Deepseek V3
# llm = ChatOpenAI(
#     model_name='deepseek-chat', 
#     openai_api_key=os.environ.get('DEEPSEEK_API_KEY'), 
#     openai_api_base='https://api.deepseek.com',
#     max_tokens=1024
# )

# Deepseek R1 Distilled
# llm = HuggingFaceHub(repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
#                     #  model_kwargs={"temperature": 0.1, "max_length": 2048, "do_sample": True},
#                      huggingfacehub_api_token=os.environ.get('HF_API_KEY')
#     )

# Cohere
llm = HuggingFaceHub(repo_id="CohereForAI/c4ai-command-r-v01",
                     huggingfacehub_api_token=os.environ.get('HF_API_KEY')
    )

# # OpenAI
# llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

def load_sys_message(file_path="system_message.txt"):
    with open(file_path, 'r') as f:
        return f.read().strip()

# Pinecone
# vectorstore = PineconeVectorStore(
#     index_name=os.environ["INDEX_NAME"], embedding=embeddings
# )
system_message = load_sys_message()
messages = [
    SystemMessage(
        content=system_message    
    ),
    HumanMessage(content="What's your purpose?"),
    AIMessage(content="What would you like to know about Jake?")
    ]

greeting = llm.invoke(messages)
# print("Greeting: ", greeting)

def response(model=llm, user_input=test_msg):
    if model == llm:
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        response = qa.run(user_input)
        print(f"Response: {response}")
    # else:
    #     response = client.text_generation(user_input)
    #     print(response)
    #     return response[0]['generated_text']

    return response

if __name__ == "__main__":
    response(model=llm)
