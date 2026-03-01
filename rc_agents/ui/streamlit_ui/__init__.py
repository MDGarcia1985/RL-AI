"""
streamlit_ui package

Streamlit UI modules for RC Agents.

Design intent:
- Keep app_streamlit.py thin
- Keep UI helpers internal unless explicitly needed
- Expose only high-level entry points
"""

from .sidebar_ui import sidebar_config
from .main_panel import render_main_panel

__all__ = [
    "sidebar_config", 
    "render_main_panel"
]