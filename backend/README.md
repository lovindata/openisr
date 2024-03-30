# Backend

## Contribution

Please install [Python](https://www.python.org/downloads/).

Please install [Poetry](https://python-poetry.org/docs/#installation) via [pipx](https://pipx.pypa.io/stable/installation/).

Please install [VSCode](https://code.visualstudio.com/) and its extensions:

- Black Formatter
- isort
- Python
- Pylance
- Even Better TOML
- Prettier

To have your Python environment inside your project (optional):

```bash
poetry config virtualenvs.in-project true
```

To create your Python environment and install dependencies:

```bash
poetry install
```

To update dependencies:

```bash
poetry update
```

To run unit tests:

```bash
pytest
```

To start the backend server:

```bash
python main.py
```

## Data migrations

To generate migration:

```bash
alembic revision --autogenerate
```

To migrate to head:

```bash
alembic upgrade head
```

To downgrade:

```bash
alembic downgrade -1
```
