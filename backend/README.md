# Backend

## Installation

Please install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

Please install VSCode extensions:

- Black Formatter
- isort
- Python
- Pylance

To install dependencies on a given `openisr` Python environnement, from `./backend` run the command:

```bash
conda create --name openisr --channel=conda-forge --file ./requirements.txt
```

Using the newly created `openisr` Python environnement from `./backend`, run [main.py](.\src\main.py).

## Data migrations

In case of data migrations for your local dev environment `./backend/data/openisr.db`, here are some commands (please adapt the path to your Alembic executable).

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
