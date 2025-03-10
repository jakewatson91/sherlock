from response import response, cancel_inference
import gradio as gr

custom_css = """
/* Light Mode (Default) */
:root {
    --bg-color: ##f5ebdf;
    --container-bg: #eBe8e6fc;
    --text-color: #333;
    --button-bg: #4CAF50;
    --button-hover-bg: #45a049;
    --input-border: #ccc;
    --shadow-color: rgba(0, 0, 0, 0.1);
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-color: #f5ebdf;
        --container-bg: #1e1e1e;
        --text-color: #ffffff;
        --button-bg: #3ea96d;
        --button-hover-bg: #2f8b59;
        --input-border: #444;
        --shadow-color: rgba(255, 255, 255, 0.1);
    }
}

/* Global Styles */
body {
    background-color: var(--bg-color);
    font-family: 'Arial', Helvetica, sans-serif;
    color: var(--text-color);
}

/* Main Container - Wider & Shorter */
.gradio-container {
    max-width: 1200px;
    margin: 20px auto;
    padding: 15px; 
    background: var(--container-bg);
    box-shadow: 0 2px 6px var(--shadow-color);
    border-radius: 8px;
}

/* Button Styling */
.gr-button {
    background-color: var(--button-bg);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.gr-button:hover {
    background-color: var(--button-hover-bg);
}

/* Input Fields (Textboxes, Chatbot, and Sliders) */
.gr-slider input, .gr-textbox input, .gr-textbox textarea, .gr-chatbot {
    color: var(--text-color);
    background-color: var(--container-bg);
    border: 1px solid var(--input-border);
}

/* Chat Messages */
.gr-chatbot {
    background: var(--container-bg);
    color: var(--text-color);
    border-radius: 10px;
    padding: 10px;
}

/* Title */
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: var(--text-color);
}
"""

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

# Define the interface
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Ask Sherlock about Jake</h1>")
    
    model_list = ["DeepSeek-V3", "Cohere Command-R", "OpenAI ChatGPT-3.5"]
    model_selection = gr.Radio(model_list, label="Model", info="Model used for inference", value=model_list[0])
    # gr.Markdown("")

    # Define a persistent state for the system message
    # system_message_state = gr.State(value=messages[0]) 

    # Parameters for model control
    # temperature = gr.Slider(minimum=0.0, maximum=4.0, value=0.7, step=0.1, label="Temperature") # change to something like intensity: https://www.gradio.app/guides/quickstart

    # Chat components
    chat_history = gr.Chatbot(label="Chat")
    user_input = gr.Textbox(show_label=False, placeholder="What would you like to know about Jake?")
    cancel_button = gr.Button("Cancel Inference", variant="danger")

    user_input.submit(response, inputs=[model_selection, user_input, chat_history], outputs=chat_history)

    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces

