"""
app_streamlit.py

Streamlit app for training and testing RC Agents.

This file is intentionally thin.
UI logic lives in rc_agents/ui/stream_lit/* modules.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import logging
import streamlit as st

# Configure logging for the application.
# This sets up the root logger once so all module loggers inherit it.
# Guarded to avoid duplicate handlers in framework environments (Streamlit may preconfigure logging).
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

st.set_page_config(page_title="RC Agents Trainer", layout="wide")
st.title("CSC370 Q-Learning Trainer (Streamlit)")

# Import AFTER Streamlit page config so reruns behave predictably.
from rc_agents.ui.streamlit_ui import sidebar_config, render_main_panel

cfg, game_type, selected_agent_ids = sidebar_config()
render_main_panel(cfg=cfg, game_type=game_type, selected_agent_ids=selected_agent_ids)