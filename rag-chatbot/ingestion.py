import os
from dotenv import load_dotenv
import pickle
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, pdf
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("ingesting data...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                                  openai_api_type=os.environ.get("OPENAI_API_KEY")
                                  )

    # load pdf document
    loader = PyPDFLoader("data/resume_data_intern_2025.pdf")
    documents = loader.load()
    
    # split entire documents into chunks  
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=25, separator='') # separator is required or else it splits by page
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
