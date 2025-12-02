"""
Echo Chamber Evaluation for News Recommendations

Measures whether recommendation algorithms create filter bubbles by comparing
the diversity of recommendations vs. the diversity of user's history.

Echo Chamber Score = diversity_of_history / diversity_of_recommendations
- Score > 1.0 = Echo chamber (recommendations MORE clustered than history)
- Score < 1.0 = Bubble breaking (recommendations MORE diverse than history)
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist
import sys
import os

# Add parent directory to path
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from recommendation_algorithms.news_corpus import NewsCorpus
from tqdm import tqdm
from recommendation_algorithms.full_run import load_mind_as_articles


@dataclass
class EchoChamberEvaluation:
    """Results for echo chamber analysis"""
    user_id: str
    history_diversity: float  # Avg pairwise distance in history
    recommendation_diversity: float  # Avg pairwise distance in recommendations
    echo_chamber_score: float  # Ratio: history_div / rec_div (>1 = echo chamber)
    clustering_effect: str  # "Echo Chamber", "Neutral", or "Bubble Breaking"


class EchoChamberAnalyzer:
    """
    Analyzes echo chamber effect using semantic clustering
    """

    def __init__(self, corpus: NewsCorpus):
        """
        Args:
            corpus: NewsCorpus with embeddings already created
        """
        self.corpus = corpus
        self.df = corpus.df
        self.embeddings = corpus.embeddings

        if self.embeddings is None:
            raise ValueError("Corpus embeddings not found. Call corpus.create_embeddings() first.")

        print(f"Loaded embeddings: {self.embeddings.shape}")

    def compute_diversity(self, indices: List[int]) -> float:
        """
        Compute average pairwise cosine distance for a set of articles

        Args:
            indices: Article indices

        Returns:
            Average pairwise distance (higher = more diverse)
        """
        if len(indices) < 2:
            return 0.0

        embeddings = self.embeddings[indices]

        # Pairwise cosine distances
        distances = pdist(embeddings, metric='cosine')

        return float(np.mean(distances))

    def evaluate_user(
        self,
        user_id: str,
        history_indices: List[int],
        recommended_indices: List[int]
    ) -> EchoChamberEvaluation:
        """
        Evaluate echo chamber effect for one user

        Args:
            user_id: User identifier
            history_indices: User's reading history
            recommended_indices: Recommended articles

        Returns:
            EchoChamberEvaluation with metrics
        """
        # Compute diversity of history
        history_diversity = self.compute_diversity(history_indices)

        # Compute diversity of recommendations
        rec_diversity = self.compute_diversity(recommended_indices)

        # Echo chamber score
        # Ratio > 1 means recs are MORE clustered (echo chamber)
        # Ratio < 1 means recs are MORE diverse (bubble breaking)
        if rec_diversity > 0:
            echo_score = history_diversity / rec_diversity
        else:
            echo_score = 1.0

        # Classify effect
        if echo_score > 1.05:
            effect = "Echo Chamber"
        elif echo_score < 0.95:
            effect = "Bubble Breaking"
        else:
            effect = "Neutral"

        return EchoChamberEvaluation(
            user_id=user_id,
            history_diversity=history_diversity,
            recommendation_diversity=rec_diversity,
            echo_chamber_score=echo_score,
            clustering_effect=effect
        )

    def evaluate_recommendations_file(
        self,
        recommendations_file: str,
        behaviors_df: pd.DataFrame,
        news_id_to_index: Dict[str, int],
        max_users: Optional[int] = None
    ) -> List[EchoChamberEvaluation]:
        """
        Evaluate all recommendations in a file

        Args:
            recommendations_file: Path to JSON file
            behaviors_df: User behaviors
            news_id_to_index: Mapping news_id to index
            max_users: Max users to evaluate (None = all)

        Returns:
            List of EchoChamberEvaluation objects
        """
        # Load recommendations
        with open(recommendations_file, 'r') as f:
            recommendations = json.load(f)

        # Create user history map
        user_history_map = {}
        for _, row in behaviors_df.iterrows():
            if pd.isna(row["history"]):
                continue

            history_ids = row["history"].split()
            user_history_indices = [
                news_id_to_index[nid]
                for nid in history_ids
                if nid in news_id_to_index
            ]

            if user_history_indices:
                user_history_map[str(row["user_id"])] = user_history_indices

        if max_users:
            recommendations = recommendations[:max_users]

        # Filter valid users
        valid_recs = [
            rec for rec in recommendations
            if rec["user_id"] in user_history_map
        ]

        print(f"Evaluating {len(valid_recs)} users for echo chamber effect...")

        evaluations = []

        for rec in tqdm(valid_recs, desc="Users"):
            user_id = rec["user_id"]
            recommended_indices = rec["recommended_indices"]
            history_indices = user_history_map[user_id]

            try:
                evaluation = self.evaluate_user(
                    user_id,
                    history_indices,
                    recommended_indices
                )
                evaluations.append(evaluation)

            except Exception as e:
                print(f"\nError on user {user_id}: {e}")
                continue

        return evaluations


def compute_statistics(evaluations: List[EchoChamberEvaluation]) -> Dict:
    """
    Compute aggregate statistics from evaluations

    Args:
        evaluations: List of EchoChamberEvaluation objects

    Returns:
        Dictionary with statistics
    """
    echo_scores = [e.echo_chamber_score for e in evaluations]
    history_divs = [e.history_diversity for e in evaluations]
    rec_divs = [e.recommendation_diversity for e in evaluations]

    # Count effects
    echo_chamber_count = sum(1 for e in evaluations if e.clustering_effect == "Echo Chamber")
    bubble_breaking_count = sum(1 for e in evaluations if e.clustering_effect == "Bubble Breaking")
    neutral_count = sum(1 for e in evaluations if e.clustering_effect == "Neutral")

    return {
        "num_users": len(evaluations),
        "echo_chamber_score": {
            "mean": float(np.mean(echo_scores)),
            "median": float(np.median(echo_scores)),
            "std": float(np.std(echo_scores)),
            "min": float(np.min(echo_scores)),
            "max": float(np.max(echo_scores))
        },
        "history_diversity": {
            "mean": float(np.mean(history_divs)),
            "std": float(np.std(history_divs))
        },
        "recommendation_diversity": {
            "mean": float(np.mean(rec_divs)),
            "std": float(np.std(rec_divs))
        },
        "effects": {
            "echo_chamber": echo_chamber_count,
            "bubble_breaking": bubble_breaking_count,
            "neutral": neutral_count,
            "echo_chamber_pct": float(echo_chamber_count / len(evaluations) * 100),
            "bubble_breaking_pct": float(bubble_breaking_count / len(evaluations) * 100)
        }
    }


def compare_methods(
    corpus: NewsCorpus,
    behaviors_df: pd.DataFrame,
    news_id_to_index: Dict[str, int],
    recommendation_files: Dict[str, str],
    max_users_per_method: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare recommendation methods on echo chamber metrics

    Args:
        corpus: NewsCorpus with embeddings
        behaviors_df: User behaviors
        news_id_to_index: ID mapping
        recommendation_files: Method name -> file path
        max_users_per_method: Max users to evaluate

    Returns:
        Comparison DataFrame
    """
    analyzer = EchoChamberAnalyzer(corpus)

    comparison_results = []
    all_evaluations = {}

    for method_name, file_path in recommendation_files.items():
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"{'='*60}")

        evaluations = analyzer.evaluate_recommendations_file(
            file_path,
            behaviors_df,
            news_id_to_index,
            max_users=max_users_per_method
        )

        all_evaluations[method_name] = evaluations

        stats = compute_statistics(evaluations)

        comparison_results.append({
            "Method": method_name,
            "N": stats["num_users"],
            "Echo_Score": f"{stats['echo_chamber_score']['mean']:.3f}",
            "Echo_Score_Std": f"{stats['echo_chamber_score']['std']:.3f}",
            "History_Div": f"{stats['history_diversity']['mean']:.3f}",
            "Rec_Div": f"{stats['recommendation_diversity']['mean']:.3f}",
            "Echo_Chamber_%": f"{stats['effects']['echo_chamber_pct']:.1f}",
            "Bubble_Breaking_%": f"{stats['effects']['bubble_breaking_pct']:.1f}",
            "Echo_Count": stats['effects']['echo_chamber'],
            "Breaking_Count": stats['effects']['bubble_breaking']
        })

        print(f"\n Echo Chamber Score: {stats['echo_chamber_score']['mean']:.3f} ± {stats['echo_chamber_score']['std']:.3f}")
        print(f"   (>1.0 = echo chamber, <1.0 = bubble breaking)")
        print(f" History Diversity:  {stats['history_diversity']['mean']:.3f}")
        print(f" Rec Diversity:      {stats['recommendation_diversity']['mean']:.3f}")
        print(f" Echo Chamber Users: {stats['effects']['echo_chamber_pct']:.1f}%")
        print(f" Bubble Breaking:    {stats['effects']['bubble_breaking_pct']:.1f}%")

    # Save detailed evaluations
    for method_name, evaluations in all_evaluations.items():
        filename = f"echo_eval_{method_name.lower().replace(' ', '_')}.json"
        eval_dicts = [
            {
                "user_id": e.user_id,
                "history_diversity": e.history_diversity,
                "recommendation_diversity": e.recommendation_diversity,
                "echo_chamber_score": e.echo_chamber_score,
                "effect": e.clustering_effect
            }
            for e in evaluations
        ]
        with open(filename, 'w') as f:
            json.dump(eval_dicts, f, indent=2)
        print(f"Saved: {filename}")

    return pd.DataFrame(comparison_results)


def main():
    import os

    # For Colab: adjust paths as needed
    MIND_PATH = "/content/MINDsmall_train"
    JSON_DIR = "/content"  # JSON files should be uploaded here

    print("="*60)
    print("ECHO CHAMBER ANALYSIS")
    print("="*60)

    print("\nLoading MIND dataset...")
    articles_df, behaviors, news_id_to_index = load_mind_as_articles(MIND_PATH)

    print("Creating corpus...")
    corpus = NewsCorpus(articles_df)
    corpus.create_embeddings()

    recommendation_files = {
        "Pure Relevance": os.path.join(JSON_DIR, "pure_relevance.json"),
        "Calibrated Diversity": os.path.join(JSON_DIR, "calibrated_diversity.json"),
        "Serendipity-Aware": os.path.join(JSON_DIR, "serendipity.json")
    }

    # Check if JSON files exist
    missing_files = [name for name, path in recommendation_files.items() if not os.path.exists(path)]
    if missing_files:
        print(f"\nWarning: Missing JSON files: {missing_files}")
        print(f"Expected location: {JSON_DIR}")
        print("Please upload the recommendation JSON files to /content/")
        return

    # Run comparison
    comparison_df = compare_methods(
        corpus,
        behaviors,
        news_id_to_index,
        recommendation_files,
        max_users_per_method=None  # None = all users
    )

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv("echo_chamber_analysis.csv", index=False)
    print("\nSaved: echo_chamber_analysis.csv")

if __name__ == "__main__":
    main()
