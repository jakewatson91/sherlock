import gradio as gr
from gradio import ChatMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from ingestion import load_embeddings
from models import deepseekv3_llm, openai_llm, llama_llm, google_gemma_llm

import warnings
warnings.filterwarnings("ignore")

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
    "OpenAI ChatGPT-3.5" : openai_llm
    }

def load_sys_message(file_path="system_message.txt"):
    with open(file_path, 'r') as f:
        return f.read().strip()

system_message = load_sys_message()

curr_model = None
curr_selection = None
def response(user_input=test_msg, model_selection="OpenAI ChatGPT-3.5", chat_history=chat_history, system_message=system_message):
    global curr_model, curr_selection
    llm = curr_model
    if not curr_model or curr_selection != model_selection: # only run if model changes
        llm = model_dict.get(model_selection)
        curr_model = llm
        curr_selection = model_selection
    print("Selected model: ", curr_selection)  

    # structured prompts for more structured models
    prompt = ChatPromptTemplate([
        ("system", "{system_message} Your name is Sherlock. {context}"),
        ("human", "{input}"),
        ]) 
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)    

    response_stream = chain.stream({"system_message": system_message, "input": user_input, "context": chat_history})

    chat_history.append(ChatMessage(role="user", content=user_input))
    chat_history.append(ChatMessage(role="assistant", content="")) # thinking

    current_text = ""
    for token in response_stream:
        answer = token.get("answer", "") # token is list of dicts - 'answer' shows up after a few iterations
        current_text += answer 
        chat_history[-1].content = current_text # Gradio expects tuples - for each user input, update with last version of response
        yield chat_history

### Define the interface ###
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Ask Sherlock about Jake</h1>")
    system_message = gr.State(system_message)
    model_list = list(model_dict.keys())
    model_selection = gr.Radio(model_list, label="Model", info="Model used for inference", value=model_list[0])

    chat_history = gr.Chatbot(label="Chat", type="messages")
    user_input = gr.Textbox(show_label=False, placeholder="What would you like to know about Jake?")
    
    examples = gr.Examples(
                    examples = [
                        ["What's Jake's experience in SQL?"],
                        ["Tell me about a Python project Jake worked on."],
                        ["How many years has Jake worked in data?"],
                        ["Give me a list of 10 technologies Jake has experience with and his level in each"]
                    ],
                    fn=response,
                    inputs=[user_input],
                    outputs=chat_history,
                    run_on_click=True
                )
    run_inference = user_input.submit(response, inputs=[user_input, model_selection, chat_history, system_message], outputs=chat_history)
    cancel_button = gr.Button("Cancel Inference", variant="danger")
    cancel_inference = cancel_button.click(fn=None, cancels=run_inference)


if __name__ == "__main__":
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces
    for chunk in response(): # for testing
        pass
    print("\n\n\n\n###################### RESPONSE ################## \n\n\n", chunk[-1].content) # final response

