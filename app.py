from stateless_bot import response
import gradio as gr

custom_css = """
#main-container {
    background-color: #fcf1db;
    font-family: 'Arial', Helvetica, sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""

def cancel_inference():
    global stop_inference
    stop_inference = True

# Define the interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Jake's experience Q&A</h1>")
    # gr.Markdown("")

    # Define a persistent state for the system message
    # system_message_state = gr.State(value=messages[0]) 

    # Parameters for model control
    # temperature = gr.Slider(minimum=0.0, maximum=4.0, value=0.7, step=0.1, label="Temperature") # change to something like intensity: https://www.gradio.app/guides/quickstart

    # Chat components
    chat_history = gr.Chatbot(label="Chat")
    user_input = gr.Textbox(show_label=False, placeholder="What would you like to know about Jake?")
    output = gr.Textbox()
    cancel_button = gr.Button("Cancel Inference", variant="danger")

    # Pass the `system_message_state` to the `response` function
    user_input.submit(response, inputs=[user_input, chat_history], outputs=output)

    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces

