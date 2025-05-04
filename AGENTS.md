# LLM Rules for Chess Engine

These rules guide how LLM-based contributors should interact with and modify this project.

## Editing these rules
If something is unclear or missing, please ask the user to clarify it. Then, modify `AGENTS.md` to reflect the new rules. If the user corrects you on a misunderstanding, update `AGENTS.md` to avoid making the same mistake again.

## Code Style

- **Imports**: Group standard library, third-party, then local imports.
- **Type Hints**: Use type hints for function parameters and return values.
- **Naming**:
  - Classes: `CamelCase`
  - Functions/Methods: `snake_case`
  - Constants: `UPPER_CASE`
- **Docstrings**: Include for modules, classes, and functions (Google or NumPy style).
- **Error Handling**: Only handle errors if needed. Do not add unnecessary error handling.
- **Modules**: Use relative imports within the package where appropriate.
- **Parameters**: For scripts, prefer UPPER_CASE constants at the top of the file as opposed to command line arguments with argparse.

## Testing

- Never run tests using `pytest` instead, use `uv run pytest`
- Run a single test: `uv run pytest tests/test_net.py::test_board_to_features -v`
- Run tests by expression: `uv run pytest -k "board_to_features"`

## Formatting

- Ensure code is formatted and linted: `uv run pre-commit run`

## Marimo Notebooks

- This codebase uses marimo notebooks instead of jupyter notebooks.
- If asked to create a notebook, look at the existing notebooks by searching the codebase for `@app.cell`
- For more examples and documentation, see `.venv/lib/python3.12/site-packages/marimo/_tutorials`

## Documentation

- See [README.md](README.md) for a project overview and quick start.
- See [docs/index.md](docs/index.md) for a list of all documentation.
