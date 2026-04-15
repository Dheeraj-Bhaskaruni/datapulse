"""Entry point for Hugging Face Spaces deployment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.gradio_app import demo  # noqa: E402

demo.launch()
