import gradio as gr
from gradio import ChatMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from ingestion import load_embeddings
from models import deepseekv3_llm, openai_llm, llama_llm, google_gemma_llm

css = """.gradio-container {
max-width: 100%;
margin: 0 auto;
}
"""

# set design theme
theme = gr.themes.Soft(
    primary_hue="stone",
    text_size="md",
    font=[gr.themes.GoogleFont('Newsreader'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

# Load embeddings once
retriever = load_embeddings()

test_msg = "Give me a quick summary of Jake's experience"

# Pinecone
# vectorstore = PineconeVectorStore(
#     index_name=os.environ["INDEX_NAME"], embedding=embeddings
# )

chat_history = []

model_dict = {
    "Llama-3.2 3B" : llama_llm,
    "Google Gemma-2 2B" : google_gemma_llm,
    "OpenAI ChatGPT-3.5" : openai_llm,
    "DeepSeek-V3" : deepseekv3_llm
    }

def load_sys_message(file_path="system_message.txt"):
    with open(file_path, 'r') as f:
        return f.read().strip()

system_message = load_sys_message()

def response(model_selection, user_input=test_msg, chat_history=chat_history, system_message=system_message):
    print("Selected model: ", model_selection)  
    llm = model_dict.get(model_selection)
    
    if model_selection in ["OpenAI ChatGPT-3.5", "DeepSeek-V3"]:
        # structured prompts for more structured models
        prompt = ChatPromptTemplate([
            ("system", "{system_message} Your name is Sherlock. {context}"),
            ("human", "{input}"),
            ]) 
    else:
        prompt = PromptTemplate(
            input_variables=["system_message", "context", "input"],
            # template="{system_message}\nYour name is Sherlock.\n{context}\n\nQuestion: {input}"
            template="""
                {system_message}\n
                ONLY use the context below to answer the question.
                Do NOT make up facts. If the answer is not in the context, say "Sorry, I don't have the answer to that."

                Context:
                {context}

                Question: {input}
                Answer:
                """
        )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)    

    response_stream = chain.stream({"system_message": system_message, "input": user_input, "context": chat_history})

    chat_history.append(ChatMessage(role="user", content=user_input))
    chat_history.append(ChatMessage(role="assistant", content="")) # thinking

    current_text = ""
    for token in response_stream:
        answer = token.get("answer", "") # token is list of dicts - 'answer' shows up after a few iterations
        current_text += answer 
        cleaned_text = current_text.split("<|eot_id|>")[0].strip() # for LLama
        chat_history[-1].content = cleaned_text # Gradio expects tuples - for each user input, update with last version of response
        yield chat_history

# Define the interface
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Ask Sherlock about Jake</h1>")
    system_message = gr.State(system_message)
    model_list = list(model_dict.keys())
    model_selection = gr.Radio(model_list, label="Model", info="Model used for inference", value=model_list[0])
    # gr.Markdown("")

    # Parameters for model control
    # temperature = gr.Slider(minimum=0.0, maximum=4.0, value=0.7, step=0.1, label="Temperature") # change to something like intensity: https://www.gradio.app/guides/quickstart

    # Chat components
    chat_history = gr.Chatbot(label="Chat", type="messages")
    user_input = gr.Textbox(show_label=False, placeholder="What would you like to know about Jake?")
    cancel_button = gr.Button("Cancel Inference", variant="danger")

    run_inference = user_input.submit(response, inputs=[model_selection, user_input, chat_history, system_message], outputs=chat_history)

    cancel_button.click(fn=None, cancels=run_inference)

if __name__ == "__main__":
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces
    for chunk in response(model_selection="Llama-3.2 3B"): # for testing
        pass
    print("\n\n\n\n###################### RESPONSE ################## \n\n\n", chunk[-1].content) # final response

