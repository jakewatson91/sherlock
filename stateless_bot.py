import os
import warnings
from dotenv import load_dotenv
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from models import deepseekv3_llm, cohere_llm, openai_llm

warnings.filterwarnings("ignore")

load_dotenv()
test_msg = "Give me a quick summary of Jake's experience"

# Convert document embedding to NumPy arra
with open('data/embedding_pairs.pkl', 'rb') as f:
    embedding_pairs = pickle.load(f)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                                openai_api_key=os.getenv("OPENAI_API_KEY")
                                )
# Create and index the document
vectorstore = FAISS.from_embeddings(embedding_pairs, embeddings)
retriever = vectorstore.as_retriever()

def load_sys_message(file_path="system_message.txt"):
    with open(file_path, 'r') as f:
        return f.read().strip()

# Pinecone
# vectorstore = PineconeVectorStore(
#     index_name=os.environ["INDEX_NAME"], embedding=embeddings
# )
system_message = load_sys_message()

chat_history = []

model_dict = {"DeepSeek-V3" : deepseekv3_llm,
                "Cohere Command-R" : cohere_llm, # TRIAL limited, swap to COHERE_API_KEY for paid
                "OpenAI ChatGPT-3.5" : openai_llm
    }

def response(model_selection, user_input=test_msg, chat_history=chat_history): 
    print("Selected model: ", model_selection)  
    llm = model_dict.get(model_selection)
    prompt = ChatPromptTemplate([
    ("system", "{system_message} Your name is Sherlock. {context}"),
    ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)    

    response = chain.invoke({"system_message": system_message, "input": user_input, "context": chat_history})
    # print(f"Response: {response['input']}")
    # print(f"Response Context: {response['context']}")
    # print(response.keys())
    print(f"Response Answer: {response['answer']}")

    history = [user_input, response["answer"]]
    chat_history.append(history)
    print(f"Chat history: {chat_history}")
    return chat_history

if __name__ == "__main__":
    response(model_selection="DeepSeek-V3") # for testing
