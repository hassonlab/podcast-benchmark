"""
Metrics package containing all evaluation metrics for the benchmark.

Metrics are organized by task type:
- embedding_metrics: Metrics for embedding/similarity tasks
- classification_metrics: Metrics for binary and multiclass classification
- regression_metrics: Metrics for continuous prediction tasks
- utils: Helper functions shared across metrics

All metrics are automatically registered with the @register_metric decorator.
"""

# Import all metric modules to register them with the decorator
from metrics import embedding_metrics
from metrics import classification_metrics
from metrics import regression_metrics
from metrics import utils
