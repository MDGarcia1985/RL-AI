"""
logger.py

Simple execution logger for RC Agents package.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from datetime import datetime
from pathlib import Path


def log_execution(event_type: str, details: str = "") -> None:
    """Log execution event to test_history.log"""
    log_file = Path(__file__).parent.parent / "data" / "test_history.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {event_type}: {details}\n")