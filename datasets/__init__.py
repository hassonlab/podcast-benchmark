"""Built-in dataset getters discovered by the benchmark runtime."""

# Importing the package is sufficient to populate the built-in registry. The
# recursive loader in main.py additionally discovers user modules placed here.
from datasets import brain_treebank, podcast  # noqa: F401
