import os
from dotenv import load_dotenv
import pickle
from langchain_community.document_loaders import PyPDFLoader
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
    loader = PyPDFLoader("rag-chatbot/data/Jake_Resume_Full_Data-W25-Intern.pdf")
    documents = loader.load()
    print(documents)
    
    # split entire documents into chunks  
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"created {len(chunks)} chunks")

    chunk_embeddings = [embeddings.embed_query(chunk if isinstance(chunk, str) else chunk.page_content) \
                        for chunk in chunks]

    # Step 4: Save embeddings and chunks locally
    with open("rag-chatbot/data/document_chunks.pkl", "wb") as f:
        pickle.dump((chunks, chunk_embeddings), f)

    print("Document embeddings saved successfully!")

    # create vector embeddings and save it in pinecone database
    # PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
