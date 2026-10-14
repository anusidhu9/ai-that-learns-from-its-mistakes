"""Hugging Face Spaces entry point: launches the MIND dashboard.

Spaces runs `python app.py` from the repository root and sets the server
address and port through environment variables that Gradio reads.
"""
from src.app import launch_app

launch_app()
