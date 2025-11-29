import numpy as np
import pandas as pd
from typing import List
from recommendation_algorithms.news_corpus import NewsCorpus

class CalibratedDiversityRecommender:
    """
    Calibrated Diversity - MMR (Maximal Marginal Relevance)

    Balances relevance with diversity using Maximal Marginal Relevance.
    Based on: Carbonell & Goldstein (1998)

   Score = λ × relevance + (1-λ) × diversity
    """
    def __init__(self, corpus: NewsCorpus, lambda_param: float = 0.7, pool_size: int = 2000):
        """
        Args:
            corpus: NewsCorpus with embeddings
            lambda_param: relevance vs diversity tradeoff
            pool_size: restrict candidate pool to top-K most relevant
        """
        self.corpus = corpus
        self.embeddings = corpus.embeddings
        self.lambda_param = lambda_param
        self.pool_size = pool_size

        # Normalize embeddings so cosine similarity becomes dot product
        self.norm_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def recommend(self, user_history: List[int], k: int = 10, exclude_history: bool = True) -> List[int]:
        if not user_history:
            return np.random.choice(len(self.embeddings), k, replace=False).tolist()

        # create user profile vector
        user_profile = np.mean(self.norm_embeddings[user_history], axis=0)

        # compute relevance using dot product
        relevance_scores = self.norm_embeddings @ user_profile

        # exclude history
        if exclude_history:
            relevance_scores[user_history] = -np.inf

        # reduce candidate pool for speedup
        candidate_pool = np.argsort(relevance_scores)[-self.pool_size:]

        # greedy MMR
        selected = []
        selected_vecs = []

        for _ in range(min(k, len(candidate_pool))):

            # Compute diversity in vectorized form
            if selected_vecs:
                # shape: (num_candidates, num_selected)
                sim_to_selected = np.dot(self.norm_embeddings[candidate_pool], np.array(selected_vecs).T)
                # max similarity for each candidate
                max_sim = sim_to_selected.max(axis=1)
                diversity = 1 - max_sim
            else:
                diversity = np.ones(len(candidate_pool))

            # relevance for candidates only
            rel_subset = relevance_scores[candidate_pool]

            # final MMR score
            mmr_scores = self.lambda_param * rel_subset + (1 - self.lambda_param) * diversity

            # pick best candidate
            best_idx_in_pool = np.argmax(mmr_scores)
            best_article = candidate_pool[best_idx_in_pool]
            selected.append(best_article)

            # store for next iteration
            selected_vecs.append(self.norm_embeddings[best_article])

            # remove from pool
            candidate_pool = np.delete(candidate_pool, best_idx_in_pool)

        return selected
