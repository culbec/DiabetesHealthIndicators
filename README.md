# DiabetesHealthIndicators

Repository containing collaborative work on predicting the diabetes risk based on collected health indicators. The ML methods are implemented from scratch and compared with existing approaches in libraries.

## Setup

This project uses `Poetry` as its package backend. For Poetry setup, refer to the [official website](https://python-poetry.org/).

To ensure robust Python version management, we use `PyEnv`. The Python version set for this project is set in the `.python-version` file. For instructions on how to setup PyEnv, refer to the [official repository](https://github.com/pyenv/pyenv).

To get you started, refer to the underlying pipeline:

- install the Poetry `shell` plugin if you do not already have it installed:

```cmd
poetry self add poetry-plugin-shell
```

- if you prefer having virtual environments created in the project root, execute this command:

```cmd
poetry config virtualenvs.in-project true
```

- install the project dependencies:

```cmd
poetry install
```

- activate the project environment:

```cmd
poetry shell
```

- run the dataset downloader:

```cmd
python ./.datasets/dwn_datasets.py
```

- setup done!
