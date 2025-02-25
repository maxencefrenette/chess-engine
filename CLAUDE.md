# Chess Engine Development Guidelines

## Commands
- **Run engine**: `uv run engine`
- **Training**:
  - Debug model: `uv run train_debug`
  - Pico model: `uv run train_pico`
- **Testing**:
  - All tests: `pytest`
  - Single test: `pytest tests/test_net.py::test_board_to_features -v`
  - Test by expression: `pytest -k "board_to_features"`

## Code Style
- **Imports**: Group standard library, then third-party, then local imports
- **Type Hints**: Use type hints for function parameters and return values
- **Naming**:
  - Classes: `CamelCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_CASE`
- **Docstrings**: Include for modules, classes, and functions
- **Lightning**: For ML models, use PyTorch Lightning with `lightning as L`
- **Error handling**: Handle errors gracefully with try/except
- **Modules**: Use relative imports within the package

## API Reference
- Chess representation: `python-chess` library
- Model framework: PyTorch + Lightning