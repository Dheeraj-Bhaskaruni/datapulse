"""WSGI entry point for cPanel deployment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.flask_app import app as application  # noqa: F401
