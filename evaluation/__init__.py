"""
Evaluation metrics and methods for recommendation algorithms.
"""

from .centroid_method import (
    compute_embedding_drift,
    analyze_embedding_drift,
    compare_recommendation_methods
)

from .llm_method import (
    LLMEvaluation,
    LLMEvaluator,
    compute_statistics,
    compare_all_methods
)

from .sentiment_method import (
    PoliticalBiasEvaluation,
    PoliticalBiasAnalyzer,
    compute_bias_statistics,
    compare_methods_bias
)

__all__ = [
    # Centroid method
    'compute_embedding_drift',
    'analyze_embedding_drift',
    'compare_recommendation_methods',
    # LLM method
    'LLMEvaluation',
    'LLMEvaluator',
    'compute_statistics',
    'compare_all_methods',
    # Sentiment method
    'PoliticalBiasEvaluation',
    'PoliticalBiasAnalyzer',
    'compute_bias_statistics',
    'compare_methods_bias',
]
