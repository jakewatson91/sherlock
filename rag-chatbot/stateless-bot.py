import os
import warnings
from dotenv import load_dotenv
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai.chat_models.base import BaseChatOpenAI

warnings.filterwarnings("ignore")

load_dotenv()

chat_history = []

if __name__ == "__main__":

    # vectorstore = PineconeVectorStore(
    #     index_name=os.environ["INDEX_NAME"], embedding=embeddings
    # )

    # chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    # Convert document embedding to NumPy array
    with open("data/document_chunks.pkl", "rb") as f:
        chunks, chunk_embeddings = pickle.load(f)
    chunk_embeddings_np = np.array([chunk_embeddings]).astype('float32')

    # Create and index the document
    vectorstore = FAISS.from_texts(chunks, embedding=chunk_embeddings_np)

    # index = faiss.IndexFlatIP(len(chunk_embeddings)) # use inner product instead of L2
    # index.add(chunk_embeddings_np)

    # Embed the user query and search
    query_embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    query_embeddings_np = np.array([query_embeddings]).astype('float32')
    chunk = vectorstore.similarity_search(query_embeddings, k=1)

    llm = BaseChatOpenAI(
        model='deepseek-chat', 
        openai_api_key='OPENAI_API_KEY', 
        openai_api_base='https://api.deepseek.com',
        max_tokens=1024
    )

    response = llm.invoke("Hi!")
    print(response.content)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    res = qa.invoke("What are Jake's top skills?")
    print(res) 

    res = qa.invoke("Tell me about Jake's experience with SQL")
    print(res)
