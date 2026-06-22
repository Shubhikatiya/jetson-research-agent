"""
Research Agent Web UI

Provides a chat interface for the research agent.
Supports text queries and image uploads.

Usage:
    python ui.py
    python ui.py --local
"""

import argparse
import asyncio
import os
from typing import Optional

import gradio as gr
from research_agent import (
    ResearchState,
    build_workflow,
    DEFAULT_OLLAMA_MODEL,
)


def build_status(step: str, message: str) -> str:
    """Build a status message with a clean prefix."""
    labels = {
        "search": "[SEARCH]",
        "extract": "[EXTRACT]",
        "analyze": "[ANALYZE]",
        "save": "[SAVE]",
        "vision": "[VISION]",
        "thinking": "[THINKING]",
        "done": "[DONE]",
        "error": "[ERROR]",
    }
    label = labels.get(step, "[AGENT]")
    return f"**{label}** {message}"


async def run_research_chat(
    message: str,
    image: Optional[str],
    local_mode: bool,
    model: str,
    history: list,
) -> list:
    """
    Run the research agent based on a chat message and optional image.
    Returns updated history for Gradio chatbot (Gradio 6.x format).
    """
    # Add user message to history (Gradio 6 format: list of dicts with role/content)
    history.append({"role": "user", "content": message})
    yield history

    # Show thinking status
    history.append({"role": "assistant", "content": build_status("thinking", "Starting research...")})
    yield history

    # Build initial state
    topic = message or "user query"
    image_path = image if image else None

    initial_state: ResearchState = {
        "topic": topic,
        "sources": [],
        "extracted_content": [],
        "analysis": "",
        "report": "",
        "iterations": 0,
        "local_mode": local_mode,
        "model": model,
        "image_path": image_path,
    }

    workflow = build_workflow()
    app = workflow.compile()

    # Run the agent
    try:
        result = await app.ainvoke(initial_state)
        report = result.get("report", "No report generated.")

        # Replace thinking message with final report
        history.pop()  # Remove thinking message
        history.append({"role": "assistant", "content": report})

    except Exception as exc:
        history.pop()  # Remove thinking message
        history.append({"role": "assistant", "content": build_status("error", str(exc))})

    yield history


def create_ui(local_mode: bool, model: str):
    """Create and launch the Gradio chat interface."""

    with gr.Blocks(title="Research Agent") as demo:
        demo.theme = gr.themes.Soft()
        gr.Markdown("# Research Agent")
        gr.Markdown("Ask a question. The agent will search the web and generate a report.")

        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="Your message",
                placeholder="Ask me to research something...",
                lines=1,
                scale=4,
            )
            image_input = gr.Image(
                label="Image (optional)",
                type="filepath",
                height=80,
                scale=1,
            )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        with gr.Row():
            mode_display = gr.Textbox(
                value="Local (Ollama)" if local_mode else "DeepSeek API",
                label="Mode",
                interactive=False,
            )
            model_display = gr.Textbox(
                value=model,
                label="Model",
                interactive=False,
            )

        async def respond(message, image, history):
            async for updated_history in run_research_chat(message, image, local_mode, model, history):
                yield updated_history

        submit_btn.click(
            respond,
            inputs=[msg_input, image_input, chatbot],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

        msg_input.submit(
            respond,
            inputs=[msg_input, image_input, chatbot],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Research Agent Web UI")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Ollama instead of DeepSeek API",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help="Ollama model name",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the UI on",
    )
    args = parser.parse_args()

    print(f"Starting Research Agent UI...")
    print(f"Mode: {'Local (Ollama)' if args.local else 'DeepSeek API'}")
    print(f"Model: {args.model}")
    print(f"Open browser at: http://localhost:{args.port}")

    demo = create_ui(local_mode=args.local, model=args.model)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
