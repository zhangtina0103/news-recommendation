import json
import numpy as np
import pandas as pd
from typing import List, Dict
from recommendation_algorithms.news_corpus import NewsCorpus
from recommendation_algorithms.full_run import load_mind_as_articles


def compute_centroid(embeddings: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Compute the centroid of embeddings for given article indices.

    Args:
        embeddings: Full embedding matrix (n_articles x embedding_dim)
        indices: List of article indices

    Returns:
        centroid vector (embedding_dim,)
    """
    if not indices:
        raise ValueError("Can't compute centroid of empty indices list")
    selected_embeddings = embeddings[indices]
    return np.mean(selected_embeddings, axis=0)


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # Compute dot product
    return np.dot(vec1_norm, vec2_norm)


def compute_embedding_drift(
    embeddings: np.ndarray,
    history_indices: List[int],
    recommended_indices: List[int]
) -> float:
    """
    Compute Embedding Drift metric for a single user.

    Embedding Drift = 1 - cos(μ_history, μ_recommended)

    High drift (close to 2) = broader exposure, diverse recommendations
    Low drift (close to 0) = echo chamber tightening, similar recommendations

    Args:
        embeddings: Full embedding matrix
        history_indices: Indices of articles in user's history
        recommended_indices: Indices of recommended articles

    Returns:
        Embedding drift score [0, 2]
    """
    # Compute centroids
    centroid_history = compute_centroid(embeddings, history_indices)
    centroid_recommended = compute_centroid(embeddings, recommended_indices)

    # Compute cosine similarity
    cos_sim = compute_cosine_similarity(centroid_history, centroid_recommended)

    # Compute drift
    drift = 1 - cos_sim

    return drift


def analyze_embedding_drift(
    recommendations_file: str,
    corpus: NewsCorpus,
    behaviors_df: pd.DataFrame,
    news_id_to_index: Dict[str, int]
) -> Dict[str, float]:
    """
    Analyze embedding drift for all users in a recommendations file.

    Args:
        recommendations_file: Path to JSON file with recommendations
        corpus: NewsCorpus object with embeddings
        behaviors_df: DataFrame with user behaviors (including history)
        news_id_to_index: Mapping from news_id to article index

    Returns:
        Dictionary with drift statistics and per-user drifts
    """
    # Load recommendations
    with open(recommendations_file, 'r') as f:
        recommendations = json.load(f)

    # Create user_id to history mapping
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

    # Compute drift for each user
    drifts = []
    user_drifts = {}

    for rec in recommendations:
        user_id = rec["user_id"]
        recommended_indices = rec["recommended_indices"]

        if user_id not in user_history_map:
            continue

        history_indices = user_history_map[user_id]

        # Compute drift
        drift = compute_embedding_drift(
            corpus.embeddings,
            history_indices,
            recommended_indices
        )

        drifts.append(drift)
        user_drifts[user_id] = drift

    # Compute statistics
    results = {
        "mean_drift": float(np.mean(drifts)),
        "median_drift": float(np.median(drifts)),
        "std_drift": float(np.std(drifts)),
        "min_drift": float(np.min(drifts)),
        "max_drift": float(np.max(drifts)),
        "num_users": len(drifts),
        "user_drifts": user_drifts
    }

    return results


def compare_recommendation_methods(
    corpus: NewsCorpus,
    behaviors_df: pd.DataFrame,
    news_id_to_index: Dict[str, int],
    recommendation_files: Dict[str, str]
) -> pd.DataFrame:
    """
    Compare embedding drift across different recommendation methods.

    Args:
        corpus: NewsCorpus object with embeddings
        behaviors_df: DataFrame with user behaviors
        news_id_to_index: Mapping from news_id to article index
        recommendation_files: Dict mapping method name to JSON file path

    Returns:
        DataFrame with comparison statistics
    """
    comparison_results = []

    for method_name, file_path in recommendation_files.items():
        results = analyze_embedding_drift(
            file_path,
            corpus,
            behaviors_df,
            news_id_to_index
        )

        comparison_results.append({
            "Method": method_name,
            "Mean Drift": results["mean_drift"],
            "Median Drift": results["median_drift"],
            "Std Drift": results["std_drift"],
            "Min Drift": results["min_drift"],
            "Max Drift": results["max_drift"],
            "Num Users": results["num_users"]
        })

        print(f"  Mean Drift: {results['mean_drift']:.4f}")
        print(f"  Median Drift: {results['median_drift']:.4f}")
        print(f"  Std Drift: {results['std_drift']:.4f}")

    return pd.DataFrame(comparison_results)

def main():
    MIND_PATH = "/content/MINDsmall_train"

    print("Loading MIND dataset...")
    articles_df, behaviors, news_id_to_index = load_mind_as_articles(MIND_PATH)

    print("Creating corpus and embeddings...")
    corpus = NewsCorpus(articles_df)
    corpus.create_embeddings()

    recommendation_files = {
        "Pure Relevance": "/content/pure_relevance.json",
        "Calibrated Diversity": "/content/calibrated_diversity.json",
        "Serendipity-Aware": "/content/serendipity.json"
    }

    # Compare methods
    comparison_df = compare_recommendation_methods(
        corpus,
        behaviors,
        news_id_to_index,
        recommendation_files
    )

    comparison_df.to_csv("embedding_drift_comparison.csv", index=False)
    print("Saved: embedding_drift_comparison.csv")

if __name__ == "__main__":
    main()
