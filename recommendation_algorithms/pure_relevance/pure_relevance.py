import numpy as np
import pandas as pd
from typing import List
from recommendation_algorithms.news_corpus import NewsCorpus
from sklearn.metrics.pairwise import cosine_similarity

class PureRelevanceRecommender:
    """
    Pure Relevance Ranking - similarity-based ranking

    Recommends articles most similar to user's reading history.
    Simple cosine similarity ranking.
    """

    def __init__(self, corpus: NewsCorpus):
        self.corpus = corpus
        self.embeddings = corpus.embeddings

    def recommend(self,
                  user_history: List[int],
                  k: int = 10,
                  exclude_history: bool = True) -> List[int]:
        """
        Get top-k recommendations based on similarity to user history

        Args:
            user_history: List of article indices the user has read
            k: Number of recommendations to return
            exclude_history: Whether to exclude already-read articles

        Returns:
            List of k article indices (recommendations)
        """
        if not user_history:
            # Cold start: return random articles
            return np.random.choice(len(self.embeddings), k, replace=False).tolist()

        # Create user profile by averaging history embeddings
        user_profile = np.mean([self.embeddings[i] for i in user_history], axis=0)

        # Calculate similarity to all articles
        similarities = cosine_similarity([user_profile], self.embeddings)[0]

        # Exclude already read articles
        if exclude_history:
            for idx in user_history:
                similarities[idx] = -np.inf

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return top_k_indices.tolist()
