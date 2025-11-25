"""
Algorithms:
1. Pure Relevance - Similarity-based ranking
2. Calibrated Diversity - MMR (Maximal Marginal Relevance)
3. Serendipity-Aware - Mix of relevant + unexpected content

Requirements:
    pip install sentence-transformers scikit-learn numpy pandas
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional


# ============================================================================
# DATA HANDLING
# ============================================================================

class NewsCorpus:
    """
    Handles news articles and their embeddings
    """
    def __init__(self, articles_df: pd.DataFrame):
        """
        Initialize with a DataFrame containing at minimum:
        - 'title': article title
        - 'content': article text

        Optional columns:
        - 'topic': category (politics, sports, etc.)
        - 'source': news source
        - 'political_leaning': liberal/conservative/neutral
        """
        self.df = articles_df
        self.embeddings = None
        self.model = None

    def create_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Create embeddings for all articles using Sentence-BERT

        Args:
            model_name: HuggingFace model name
                - 'all-MiniLM-L6-v2': Fast, 384-dim (recommended)
                - 'all-mpnet-base-v2': Slower, 768-dim, better quality
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

        print("Creating embeddings...")
        texts = (self.df['title'] + ' ' + self.df['content']).tolist()
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        print(f"Created embeddings: {self.embeddings.shape}")
        return self.embeddings

    def save_embeddings(self, filepath: str):
        """Save embeddings to disk"""
        np.save(filepath, self.embeddings)
        print(f"Saved embeddings to {filepath}")

    def load_embeddings(self, filepath: str):
        """Load pre-computed embeddings"""
        self.embeddings = np.load(filepath)
        print(f"Loaded embeddings: {self.embeddings.shape}")
        return self.embeddings


# ============================================================================
# RECOMMENDATION ALGORITHMS
# ============================================================================

class PureRelevanceRecommender:
    """
    Algorithm 1: Pure Relevance Ranking

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


class CalibratedDiversityRecommender:
    """
    Algorithm 2: Calibrated Diversity (MMR)

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

        # === Step 1: User profile vector ===
        user_profile = np.mean(self.norm_embeddings[user_history], axis=0)

        # === Step 2: Compute relevance once (vectorized) ===
        relevance_scores = self.norm_embeddings @ user_profile

        # Exclude history
        if exclude_history:
            relevance_scores[user_history] = -np.inf

        # === Step 3: Reduce candidate pool for diversity (massive speedup) ===
        candidate_pool = np.argsort(relevance_scores)[-self.pool_size:]

        # === Step 4: Greedy MMR ===
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


class SerendipityAwareRecommender:
    """
    Algorithm 3: Serendipity-Aware Recommendations

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
