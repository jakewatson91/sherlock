import os
from dotenv import load_dotenv
import pickle
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, pdf
from langchain_community.document_loaders.rtf import UnstructuredRTFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest(file="data/experience_for_sherlock.rtf"):
    print("ingesting data...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                                  openai_api_type=os.environ.get("OPENAI_API_KEY")
                                  )
    # load pdf document
    # loader = PyPDFLoader("data/experience_for_sherlock.rtf")
    loader = UnstructuredRTFLoader(file)
    documents = loader.load()
    
    # split entire documents into chunks  
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator='') # separator is required or else it splits by page
    texts = text_splitter.split_documents(documents)

    text_embeddings = embeddings.embed_documents([text.page_content for text in texts])
    text_embeddings = np.array(text_embeddings).astype('float32')
    embedding_pairs = [(text.page_content, text_embedding) for text, text_embedding in zip(texts, text_embeddings)]

    # Step 4: Save embeddings and chunks locally
    with open("data/embedding_pairs.pkl", "wb") as f:
        pickle.dump((embedding_pairs), f)

    print(f"{len(embedding_pairs)} document embedding pairs saved successfully!")

    # create vector embeddings and save it in pinecone database
    # PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))

def load_embeddings():
    with open("data/embedding_pairs.pkl", "rb") as f:
        embedding_pairs = pickle.load(f)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_embeddings(embedding_pairs, embeddings)
    retriever = vectorstore.as_retriever()

    return retriever

if __name__ == "__main__":
    ingest()

