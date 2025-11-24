"""
Three Recommendation Algorithms for News Feed Project

Just the core recommender implementations - no evaluation framework.
Use these with your own evaluation setup.

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

    def __init__(self, corpus: NewsCorpus, lambda_param: float = 0.7):
        """
        Args:
            corpus: NewsCorpus with embeddings
            lambda_param: Trade-off between relevance and diversity
                - 1.0 = pure relevance
                - 0.0 = pure diversity
                - 0.7 = recommended (70% relevance, 30% diversity)
        """
        self.corpus = corpus
        self.embeddings = corpus.embeddings
        self.lambda_param = lambda_param

    def recommend(self,
                  user_history: List[int],
                  k: int = 10,
                  exclude_history: bool = True) -> List[int]:
        """
        Get top-k recommendations using MMR

        Args:
            user_history: List of article indices the user has read
            k: Number of recommendations to return
            exclude_history: Whether to exclude already-read articles

        Returns:
            List of k article indices (recommendations)
        """
        if not user_history:
            return np.random.choice(len(self.embeddings), k, replace=False).tolist()

        # User profile
        user_profile = np.mean([self.embeddings[i] for i in user_history], axis=0)

        # Candidate pool
        candidates = list(range(len(self.embeddings)))
        if exclude_history:
            candidates = [i for i in candidates if i not in user_history]

        # Iteratively select articles
        selected = []

        for _ in range(min(k, len(candidates))):
            mmr_scores = []

            for candidate_idx in candidates:
                # Relevance: similarity to user profile
                relevance = cosine_similarity(
                    [user_profile],
                    [self.embeddings[candidate_idx]]
                )[0][0]

                # Diversity: distance from already selected articles
                if selected:
                    # Find maximum similarity to any selected article
                    similarities_to_selected = [
                        cosine_similarity(
                            [self.embeddings[candidate_idx]],
                            [self.embeddings[s]]
                        )[0][0]
                        for s in selected
                    ]
                    max_similarity = max(similarities_to_selected)
                    diversity = 1 - max_similarity
                else:
                    diversity = 1.0  # First article has max diversity

                # MMR score: weighted combination
                mmr = self.lambda_param * relevance + (1 - self.lambda_param) * diversity
                mmr_scores.append(mmr)

            # Select article with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(candidates[best_idx])
            candidates.pop(best_idx)

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


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("="*70)
    print("News Recommendation Algorithms - Demo")
    print("="*70)

    # Create sample data
    print("\n1. Creating sample news corpus...")

    titles = [
        'Biden announces new climate policy',
        'Trump rally draws thousands',
        'Stock market hits record high',
        'Lakers win championship',
        'New AI model released',
        'Supreme Court ruling issued',
        'Tech startup raises funding',
        'Hollywood strike continues',
        'Congress passes spending bill',
        'Scientists discover new species',
    ]

    contents = [
        'The administration announced sweeping climate initiatives targeting emissions...',
        'Former president addresses supporters at packed stadium venue...',
        'Wall Street celebrates as major indices reach unprecedented levels...',
        'Basketball team clinches title in dramatic final game...',
        'Research team unveils breakthrough artificial intelligence system...',
        'High court makes landmark decision affecting national policy...',
        'Silicon Valley company secures major venture capital investment...',
        'Entertainment industry labor dispute enters critical phase...',
        'Legislature approves budget after heated negotiations...',
        'Marine biologists announce remarkable deep sea finding...',
    ]

    # Repeat to create more articles
    sample_articles = pd.DataFrame({
        'title': titles * 10,
        'content': contents * 10
    })

    print(f"Created corpus with {len(sample_articles)} articles")

    # Create embeddings
    print("\n2. Creating embeddings...")
    corpus = NewsCorpus(sample_articles)
    corpus.create_embeddings()

    # Initialize recommenders
    print("\n3. Initializing recommenders...")
    rec1 = PureRelevanceRecommender(corpus)
    rec2 = CalibratedDiversityRecommender(corpus, lambda_param=0.7)
    rec3 = SerendipityAwareRecommender(corpus, serendipity_ratio=0.3)

    print("✓ Pure Relevance")
    print("✓ Calibrated Diversity (λ=0.7)")
    print("✓ Serendipity-Aware (30% unexpected)")

    # Simulate user with history
    print("\n4. Simulating user with reading history...")
    user_history = [0, 1, 5, 8]  # Read some political articles
    print("User has read articles:", user_history)

    # Get recommendations from each algorithm
    print("\n5. Getting recommendations (k=10)...")

    recs1 = rec1.recommend(user_history, k=10)
    recs2 = rec2.recommend(user_history, k=10)
    recs3 = rec3.recommend(user_history, k=10)

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print("\n📊 Pure Relevance:")
    for i, idx in enumerate(recs1[:5], 1):
        print(f"  {i}. [{idx}] {corpus.df.iloc[idx]['title']}")

    print("\n🎯 Calibrated Diversity (MMR):")
    for i, idx in enumerate(recs2[:5], 1):
        print(f"  {i}. [{idx}] {corpus.df.iloc[idx]['title']}")

    print("\n✨ Serendipity-Aware:")
    for i, idx in enumerate(recs3[:5], 1):
        print(f"  {i}. [{idx}] {corpus.df.iloc[idx]['title']}")

    print("\n" + "="*70)
    print("Demo complete!")
    print("\nNow use these recommenders with your own evaluation framework.")
    print("="*70)
