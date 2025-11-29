import numpy as np
import pandas as pd
from typing import List
from recommendation_algorithms.news_corpus import NewsCorpus
from sklearn.metrics.pairwise import cosine_similarity


class SerendipityAwareRecommender:
    """
    Serendipity-Aware Recommendations

    Deliberately includes unexpected content from different topics/viewpoints.
    Inspired by xQuAD approach (Vargas et al. 2014)

    Strategy: Mix relevant articles with distant (but quality) content
    """

    def __init__(self, corpus: NewsCorpus, serendipity_ratio: float = 0.3):
        """
        Args:
            corpus: NewsCorpus with embeddings
            serendipity_ratio: Fraction of recommendations that are "unexpected"
                - 0.1 = 10% serendipitous
                - 0.3 = 30% serendipitous (recommended)
                - 0.5 = 50% serendipitous
        """
        self.corpus = corpus
        self.embeddings = corpus.embeddings
        self.serendipity_ratio = serendipity_ratio

    def recommend(self,
                  user_history: List[int],
                  k: int = 10,
                  exclude_history: bool = True) -> List[int]:
        """
        Get top-k recommendations with serendipitous content

        Args:
            user_history: List of article indices the user has read
            k: Number of recommendations to return
            exclude_history: Whether to exclude already-read articles

        Returns:
            List of k article indices (recommendations)
        """
        if not user_history:
            return np.random.choice(len(self.embeddings), k, replace=False).tolist()

        # Split recommendations
        n_relevant = int(k * (1 - self.serendipity_ratio))
        n_serendipitous = k - n_relevant

        # User profile
        user_profile = np.mean([self.embeddings[i] for i in user_history], axis=0)

        # === RELEVANT ARTICLES ===
        # Get top similar articles
        similarities = cosine_similarity([user_profile], self.embeddings)[0]
        if exclude_history:
            for idx in user_history:
                similarities[idx] = -np.inf

        relevant_indices = np.argsort(similarities)[-n_relevant:][::-1].tolist()

        # === SERENDIPITOUS ARTICLES ===
        # Get articles that are distant from user profile
        distances = 1 - similarities

        # Exclude relevant articles and history
        available = [
            i for i in range(len(self.embeddings))
            if i not in relevant_indices and (not exclude_history or i not in user_history)
        ]

        if available:
            # Select diverse articles (high distance from user profile)
            distances_available = distances[available]
            serendipitous_sorted = sorted(
                zip(available, distances_available),
                key=lambda x: x[1],
                reverse=True
            )
            serendipitous_indices = [idx for idx, _ in serendipitous_sorted[:n_serendipitous]]
        else:
            serendipitous_indices = []

        # Combine and shuffle (so serendipitous items aren't all at the end)
        recommendations = relevant_indices + serendipitous_indices
        np.random.shuffle(recommendations)

        return recommendations
