import gradio as gr
import subprocess
import queue
import threading

def run_training():
    # Start the training script as a subprocess
    process = subprocess.Popen(
        ["bash", "launch_grpo.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Yield logs line-by-line to the Gradio UI
    for line in iter(process.stdout.readline, ''):
        yield line

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 AxiomForge-RL GRPO Training")
    gr.Markdown("Click the button below to start the Qwen-1.5B competitive math GRPO pipeline.")
    
    start_btn = gr.Button("Start GRPO Training", variant="primary")
    log_output = gr.Textbox(label="Training Logs", lines=30, max_lines=50)
    
    start_btn.click(fn=run_training, outputs=log_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
