Logging Architecture

This project uses Python’s built-in logging module for structured runtime diagnostics.

The design follows a strict separation of responsibilities:

Agents and library modules define loggers

Application entry points configure logging

Logging is configured once per process

Design Philosophy

We treat logging as an application concern, not a library concern.

Agents (rl_agent.py, rlf_agent.py, q_agent.py, etc.) never configure logging.

Entry points (`__main__.py` for CLI, `app_streamlit.py` for Streamlit) configure logging.

Logging is centralized and inherited by all modules.

This prevents:

Duplicate log handlers

Conflicting formats

Overwritten logging levels

Framework interference

How Logging Works in This Project
1. Modules define loggers

Every module that needs logging does:

import logging
logger = logging.getLogger(__name__)

This creates a named logger tied to the module’s import path, for example:

rc_agents.edge_ai.rcg_edge.agents.rlf_agent

These loggers do not configure anything. They simply emit messages.

Example usage inside an agent:

logger.warning("Invalid weights in _action_from_theta_soft: w=%s, s=%s", w, s)
2. Entry points configure logging

Logging configuration belongs in application entry points only.

In __main__.py:

import logging

# Configure logging for the application.
# This sets up the root logger once so all module loggers inherit it.
# Guarded to avoid duplicate handlers in framework environments.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

This configures:

Log level

Output format

Root handler

All module loggers propagate to this root logger.

Why the Guard Is Used
if not logging.getLogger().hasHandlers():

This ensures logging is configured only if it hasn’t already been configured.

It is important in environments like:

Streamlit

Jupyter notebooks

Other frameworks that may preconfigure logging

Without this guard, duplicate log lines can occur.

Where Logging Should and Should Not Be Configured
Correct locations: `__main__.py` (CLI), `app_streamlit.py` (Streamlit). Only these application entry points should call `logging.basicConfig(...)`.

Incorrect Locations

rl_agent.py

rlf_agent.py

q_agent.py

utils/

Any reusable module

Libraries should never call:

logging.basicConfig(...)

Only applications configure logging.

Log Levels

Common levels used in this project:

logger.debug(...) — detailed internal state tracing

logger.info(...) — normal operational milestones

logger.warning(...) — recoverable anomalies

logger.error(...) — serious issues

Default level is set to:

level=logging.INFO

Change to DEBUG during development if deeper trace is required.

Example Output Format
2026-02-21 14:22:01,342 [WARNING] rc_agents.edge_ai.rcg_edge.agents.rlf_agent: Invalid weights in _action_from_theta_soft: w=[...], s=0.0

Format components:

Timestamp

Level

Module name

Message

Important Warning

Do not create a file named:

logging.py

inside this project.

Doing so will shadow Python’s built-in logging module and break imports.

Summary

Modules define loggers.

Entry points configure logging once.

Use guard in framework environments.

Do not configure logging inside agents.

Do not shadow the standard library.

This structure ensures:

Clean separation of concerns

Predictable logging behavior

Framework compatibility

Production-ready diagnostics

End of logging specification.