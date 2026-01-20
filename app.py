import chainlit as cl
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import snapshot_download
from langchain.tools import tool
from langchain_groq import ChatGroq
from typing import cast
import os

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi

# 1. Is the key actually there?
print(f"DEBUG - Token exists: {bool(os.environ.get('HF_TOKEN'))}")

# 2. Does HF recognize this key?
try:
    print(f"DEBUG - Auth check: {HfApi(token=os.environ.get('HF_TOKEN')).whoami()['name']}")
except Exception as e:
    print(f"DEBUG - Auth failed: {e}")


from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = InMemoryVectorStore(embedding_model)


def load_sys_prompt(file_path="system_message.txt"):
    with open(file_path, 'r') as f:
        return f.read().strip()

system_prompt = load_sys_prompt()

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> str:
    """Retrieve information to help answer a query."""

    print(f"DEBUG: Tool called with query: {query}") # See this in your terminal
    retrieved_docs = vector_store.similarity_search(query, k=2)

    if not retrieved_docs:
        print("DEBUG: No documents found in vector store!")
        return "No relevant documents found.", []
    
    print(f"{len(retrieved_docs)} document(s) found in vector store.")
    print(f"First document: {retrieved_docs[0].page_content[:50]}...")

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@cl.on_chat_start
async def on_chat_start():

    try:
    # download rag data files from private HF dataset
        snapshot_download(
            repo_id="jakewatson91/sherlock-rag-docs",
            repo_type="dataset",
            local_dir="data/",
            allow_patterns="*.pdf",
            token=True
        )
    except Exception as e:
        print(f"Error downloading data: {e}")

    loader = DirectoryLoader(
        "data/",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(all_splits)

    llm = init_chat_model(
        model="moonshotai/kimi-k2-instruct-0905", 
        model_provider="groq",
        streaming=True,
        temperature=0
        )
    
    runnable = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt
    )

    cl.user_session.set("runnable", runnable)
    cl.user_session.set("memory", [])

# Send a response back to the user
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    res = cl.Message(content="")
    cb = cl.LangchainCallbackHandler()

    memory = cl.user_session.get("memory")
    memory.append({"role": "user", "content": message.content})

    async for msg, metadata in runnable.astream(
        {"messages": memory},
        stream_mode="messages",
        config=RunnableConfig(
            callbacks=[cb],
            run_name="Sherlock Search")
            ):

        if (
            msg.content 
            and metadata.get("langgraph_node") == "model" 
            and not getattr(msg, "tool_calls", None)
            ):

            print("METADATA: ", metadata)
            print("MESSAGE: ", msg)
            
            await res.stream_token(msg.content)

    memory.append({"role": "assistant", "content": res.content})
    cl.user_session.set("memory", memory)

    await res.send()

if __name__ == "__main__":
    snapshot_download(
    repo_id="jakewatson91/sherlock-rag-docs",
    repo_type="dataset",
    local_dir="data/",
    allow_patterns="*.pdf",
    token=os.environ.get("HF_TOKEN")
)