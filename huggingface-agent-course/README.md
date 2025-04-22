# Hugging Face Agent Course

This project is designed to implement a Hugging Face agent using Python. It includes various modules for handling data, utilities, and the main agent logic.

## Project Structure

```
huggingface-agent-course
├── src
│   ├── agent
│   │   ├── __init__.py
│   │   └── main.py
│   ├── data
│   │   └── __init__.py
│   └── utils
│       └── __init__.py
├── tests
│   └── __init__.py
├── .env
├── .gitignore
├── pyproject.toml
├── .ruff.toml
├── mypy.ini
├── README.md
└── requirements.txt
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd huggingface-agent-course
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables in the `.env` file. Ensure that sensitive information such as API tokens and keys are not exposed in the codebase.

## Usage

To run the main agent, execute the following command:
```
python src/agent/main.py
```

## Testing

To run the tests, ensure you have the testing framework installed and execute:
```
pytest tests/
```

## Linting and Type Checking

This project uses `ruff` for linting and `mypy` for type checking. You can run them using:
```
ruff check .
mypy src/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.