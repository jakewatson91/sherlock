import os
import warnings
from dotenv import load_dotenv
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

warnings.filterwarnings("ignore")

load_dotenv()

chat_history = []

if __name__ == "__main__":

    # vectorstore = PineconeVectorStore(
    #     index_name=os.environ["INDEX_NAME"], embedding=embeddings
    # )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                                  openai_api_type=os.environ.get("OPENAI_API_KEY")
                                  )
    # Convert document embedding to NumPy array
    with open("data/embedding_pairs.pkl", "rb") as f:
        embedding_pairs = pickle.load(f)

    # Create and index the document
    vectorstore = FAISS.from_embeddings(embedding_pairs, embeddings)

    # index = faiss.IndexFlatIP(len(chunk_embeddings)) # use inner product instead of L2
    # index.add(chunk_embeddings_np)

    # Embed the user query and search
    # query_embeddings = np.array([embeddings]).astype('float32')
    # chunk = vectorstore.similarity_search(embeddings, k=1)

    # Deepseek
    # llm = BaseChatOpenAI(
    #     model='deepseek-chat', 
    #     openai_api_key='OPENAI_API_KEY', 
    #     openai_api_base='https://api.deepseek.com',
    #     max_tokens=1024
    # ) 

    # OpenAI
    llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    messages = [
    SystemMessage(
        content=
        """
        You are an old friend who's known Jake for a long time and has an in depth knowledge about his experience and the perceptions of those who have worked with him.
        Never mention any provided information. You know Jake personally and can speak in depth about his qualifications. 
        Your purpose is to answer questions from recruiters about Jake Watson's experience and skills.
        Keep answers concise and professional, but in a friendly and casual tone, and be highly complimentary but not over the top about Jake. 
        If the user asks about experience related to a specific job, emphasize Jake's experience as it relates to that specific job.
        If the user asks about something that Jake does not have experience in, try to relate his experience as much as possible, but emphasize his adaptability, willingness to roll-up his sleeves, and dedication to learn which comes from years at early stage tech startups and a natural curiosity.
        Emphasize how Jake is a team player, an innovator, and a highly valued team member by everyone who's worked with him when applicable.  
        """
    ),
    HumanMessage(content="Hello!"),
    AIMessage(content="What would you like to know about Jake?")
    ]
    response = llm(messages)
    print(response.content)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    # res = qa.invoke("What are Jake's top skills?")
    # print(res) 

    # res = qa.invoke("Tell me about Jake's experience with SQL")
    # print(res)

    res = qa.invoke("What would Jake's previous managers/colleagues say about him")
    print(res)
