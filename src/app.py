import gradio as gr
from chatbot.chatbot_backend import ChatBot
from utils.ui_settings import UISettings

css = """
#chatbot {
    height: 500px;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    padding: 10px;
    background: #FAFAFA;
}
#input_box textarea {
    font-size: 16px !important;
    line-height: 1.5 !important;
    border-radius: 10px !important;
    padding: 10px !important;
    border: 1px solid #D1D5DB !important;
}
.gr-button {
    border-radius: 10px !important;
    font-weight: 500 !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.TabItem("HybridChatbot"):
            ##############
            # First ROW: Chat Display
            ##############
            with gr.Row():
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    avatar_images=("images/user.png", "images/chatbot.png"),
                )
                chatbot.like(UISettings.feedback, None, None)

            ##############
            # SECOND ROW: Input
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=3,
                    scale=8,
                    show_label=False,
                    placeholder="Type your message... (Enter to send, Shift+Enter for new line)",
                    container=False,
                    elem_id="input_box"
                )

            ##############
            # Third ROW: Buttons
            ##############
            with gr.Row():
                text_submit_btn = gr.Button(value="Send", elem_classes="send-btn")
                clear_button = gr.ClearButton([input_txt, chatbot], value="Clear Chat")

            ##############
            # Logic
            ##############
            # Submit on Enter
            txt_msg = input_txt.submit(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False
            ).then(
                lambda: gr.Textbox(interactive=True),
                None, [input_txt], queue=False
            )

            # Submit on button click
            txt_msg = text_submit_btn.click(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt],
                outputs=[input_txt, chatbot],
                queue=False
            ).then(
                lambda: gr.Textbox(interactive=True),
                None, [input_txt], queue=False
            )

if __name__ == "__main__":
    demo.launch()