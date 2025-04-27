# LLM Rules for Chess Engine

These rules guide how LLM-based contributors should interact with and modify this project.

## Editing these rules
To edit these rules, use the `llm-rules.md` file in the root directory of the project. If something is unclear or missing, please ask the user to clarify it. Then, modify `llm-rules.md` to reflect the new rules.

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

- Run the full test suite: `uv run pytest`
- Run a single test: `uv run pytest tests/test_net.py::test_board_to_features -v`
- Run tests by expression: `uv run pytest -k "board_to_features"`

## Formatting

- Ensure code is formatted and linted: `uv run pre-commit run`

## Documentation

- See [README.md](README.md) for a project overview and quick start.
- See [docs/index.md](docs/index.md) for a list of all documentation.
