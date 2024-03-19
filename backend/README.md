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

Using the newly created `openisr` Python environnement from `./backend`, run [main.py](.\src\main.py) (please adapt the paths):

```bash
& D:/prog/miniconda/envs/openisr/python.exe d:/prog/proj/openisr/backend/src/main.py
```

## Data migrations

In case of data migrations for your local dev environment `./backend/data/openisr.db`, here are some commands (please adapt the paths).

- To generate migration:

```bash
D:\prog\miniconda\envs\openisr\Scripts\alembic.exe revision --autogenerate
```

- To migrate to head:

```bash
D:\prog\miniconda\envs\openisr\Scripts\alembic.exe upgrade head
```

- To downgrade:

```bash
D:\prog\miniconda\envs\openisr\Scripts\alembic.exe downgrade -1
```
