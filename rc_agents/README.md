ACTION_DELTAS = {
    ACTION_FORWARD: (-1, 0),
    ACTION_BACKWARD: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_NORTHWEST: (-1, -1),
    ACTION_NORTHEAST: (-1, 1),
    ACTION_SOUTHWEST: (1, -1),
    ACTION_SOUTHEAST: (1, 1),
}

d_row, d_col = ACTION_DELTAS[action]
row += d_row
col += d_col

## Coding Conventions

This project follows standard Python naming conventions to clarify semantic roles:

- **CamelCase** for classes and type aliases
- **snake_case** for functions, methods, and variables
- **ALL_CAPS** for constants and enum-like values

These conventions are used consistently across environments, agents, runners,
and UI code to improve readability and reduce ambiguity.

These conventions are documented explicitly to support readability for students and contributors who may be new to larger Python codebases.

https://docs.pytest.org/en/stable/