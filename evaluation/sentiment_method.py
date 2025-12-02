"""
Political Bias Analysis for News Recommendations

For Google Colab:
1. Clone repo: !git clone <repo_url>
2. Upload JSON recommendation files to /content/ (or adjust JSON_DIR in main())
3. Run: python evaluation/sentiment_method.py
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import sys
import os

# Add parent directory to path for imports (works when running as script or when package not installed)
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from recommendation_algorithms.news_corpus import NewsCorpus
from tqdm import tqdm
from recommendation_algorithms.full_run import load_mind_as_articles


@dataclass
class PoliticalBiasEvaluation:
    """Results for political bias analysis"""
    user_id: str
    history_mean_bias: float  # -1 (liberal) to +1 (conservative)
    recommended_mean_bias: float
    bias_shift: float  # How much recommendations shift user
    bias_diversity: float  # Std dev of recommended bias scores
    history_scores: List[float]  # Individual article scores
    recommended_scores: List[float]


class PoliticalBiasAnalyzer:
    """
    Analyzes political bias/leaning of news articles using HuggingFace models
    """

    def __init__(
        self,
        corpus: NewsCorpus,
        model_name: str = "valurank/distilroberta-bias",
        use_gpu: bool = True
    ):
        """
        Args:
            corpus: NewsCorpus with article data
            model_name: HuggingFace model for bias detection
                Options:
                - "valurank/distilroberta-bias" (trained on news bias)
                - "cardiffnlp/twitter-roberta-base-sentiment-latest" (sentiment)
        """
        self.corpus = corpus
        self.model_name = model_name
        self.df = corpus.df
        self.use_gpu = use_gpu
        self.classifier = None

    def _load_classifier(self):
        """Load HuggingFace classifier"""
        if self.classifier is not None:
            return

        try:
            from transformers import pipeline
            import torch

            print(f"Loading bias classifier: {self.model_name}")

            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                device = 0
            else:
                device = -1

            # Load classifier
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=device,
                truncation=True,
                max_length=512
            )

            print("Classifier loaded.\n")

        except ImportError as e:
            print("ERROR: Install with: pip install transformers torch")
            raise e

    def _score_to_spectrum(self, result: Dict) -> float:
        """
        Convert classifier output to -1 (liberal) to +1 (conservative) spectrum

        Args:
            result: Classifier output dict with 'label' and 'score'

        Returns:
            Score on political spectrum (-1 to +1)
        """
        label = result['label'].lower()
        confidence = result['score']

        # Map labels to spectrum
        if 'bias' in self.model_name.lower() or 'valurank' in self.model_name.lower():
            # valurank/distilroberta-bias labels: left, center, right
            if 'left' in label or 'liberal' in label:
                return -confidence  # Negative = liberal
            elif 'right' in label or 'conserv' in label:
                return confidence  # Positive = conservative
            else:  # center/neutral
                return 0.0

        elif 'sentiment' in self.model_name.lower():
            # Sentiment models: positive, negative, neutral
            # This is a rough proxy - sentiment != political bias
            if 'positive' in label:
                return confidence * 0.3  # Weak signal
            elif 'negative' in label:
                return -confidence * 0.3
            else:
                return 0.0

        else:
            # Default: try to infer from label
            if any(word in label for word in ['left', 'liberal', 'progressive', 'negative']):
                return -confidence
            elif any(word in label for word in ['right', 'conservative', 'positive']):
                return confidence
            else:
                return 0.0

    def classify_article(self, article_index: int) -> float:
        """
        Classify single article on political spectrum

        Args:
            article_index: Index in corpus

        Returns:
            Bias score: -1 (liberal) to +1 (conservative)
        """
        self._load_classifier()

        # Get article text
        title = self.df.iloc[article_index]['title']
        content = self.df.iloc[article_index]['content']

        # Combine title and content (truncate if too long)
        text = f"{title}. {content}"[:512]

        # Classify
        result = self.classifier(text)[0]

        # Convert to spectrum score
        score = self._score_to_spectrum(result)

        return score

    def classify_articles_batch(
        self,
        article_indices: List[int],
        batch_size: int = 16
    ) -> List[float]:
        """
        Classify multiple articles in batches (faster!)

        Args:
            article_indices: List of article indices
            batch_size: Batch size for processing

        Returns:
            List of bias scores
        """
        self._load_classifier()

        scores = []

        for i in tqdm(range(0, len(article_indices), batch_size), desc="Classifying"):
            batch_indices = article_indices[i:i+batch_size]

            # Prepare texts
            texts = []
            for idx in batch_indices:
                title = self.df.iloc[idx]['title']
                content = self.df.iloc[idx]['content']
                text = f"{title}. {content}"[:512]
                texts.append(text)

            # Classify batch
            results = self.classifier(texts)

            # Convert to spectrum scores
            batch_scores = [self._score_to_spectrum(r) for r in results]
            scores.extend(batch_scores)

        return scores

    def evaluate_user_recommendations(
        self,
        user_id: str,
        history_indices: List[int],
        recommended_indices: List[int]
    ) -> PoliticalBiasEvaluation:
        """
        Evaluate political bias diversity for one user

        Args:
            user_id: User identifier
            history_indices: User's reading history
            recommended_indices: Recommended articles

        Returns:
            PoliticalBiasEvaluation with metrics
        """
        # Classify history
        history_scores = self.classify_articles_batch(history_indices)

        # Classify recommendations
        rec_scores = self.classify_articles_batch(recommended_indices)

        # Compute metrics
        history_mean = float(np.mean(history_scores))
        rec_mean = float(np.mean(rec_scores))
        bias_shift = rec_mean - history_mean
        bias_diversity = float(np.std(rec_scores))

        return PoliticalBiasEvaluation(
            user_id=user_id,
            history_mean_bias=history_mean,
            recommended_mean_bias=rec_mean,
            bias_shift=bias_shift,
            bias_diversity=bias_diversity,
            history_scores=history_scores,
            recommended_scores=rec_scores
        )

    def evaluate_recommendations_file(
        self,
        recommendations_file: str,
        behaviors_df: pd.DataFrame,
        news_id_to_index: Dict[str, int],
        max_users: Optional[int] = None
    ) -> List[PoliticalBiasEvaluation]:
        """
        Evaluate all users in a recommendations file

        Args:
            recommendations_file: Path to JSON recommendations
            behaviors_df: User behaviors DataFrame
            news_id_to_index: Mapping news_id to index
            max_users: Max users to evaluate (None = all)

        Returns:
            List of PoliticalBiasEvaluation objects
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

        print(f"Evaluating {len(valid_recs)} users for political bias...")

        evaluations = []

        for rec in tqdm(valid_recs, desc="Users"):
            user_id = rec["user_id"]
            recommended_indices = rec["recommended_indices"]
            history_indices = user_history_map[user_id]

            try:
                evaluation = self.evaluate_user_recommendations(
                    user_id,
                    history_indices,
                    recommended_indices
                )
                evaluations.append(evaluation)

            except Exception as e:
                print(f"\nError on user {user_id}: {e}")
                continue

        return evaluations


def compute_bias_statistics(evaluations: List[PoliticalBiasEvaluation]) -> Dict:
    """
    Compute aggregate statistics from bias evaluations

    Args:
        evaluations: List of PoliticalBiasEvaluation objects

    Returns:
        Dictionary with statistics
    """
    history_means = [e.history_mean_bias for e in evaluations]
    rec_means = [e.recommended_mean_bias for e in evaluations]
    bias_shifts = [e.bias_shift for e in evaluations]
    diversities = [e.bias_diversity for e in evaluations]

    # Count users by bias direction
    liberal_history = sum(1 for x in history_means if x < -0.1)
    conservative_history = sum(1 for x in history_means if x > 0.1)
    neutral_history = len(history_means) - liberal_history - conservative_history

    # Shifts
    shifted_more_liberal = sum(1 for x in bias_shifts if x < -0.05)
    shifted_more_conservative = sum(1 for x in bias_shifts if x > 0.05)
    no_shift = len(bias_shifts) - shifted_more_liberal - shifted_more_conservative

    stats = {
        "num_users": len(evaluations),

        "history_bias": {
            "mean": float(np.mean(history_means)),
            "std": float(np.std(history_means)),
            "liberal_users": liberal_history,
            "conservative_users": conservative_history,
            "neutral_users": neutral_history
        },

        "recommended_bias": {
            "mean": float(np.mean(rec_means)),
            "std": float(np.std(rec_means))
        },

        "bias_shift": {
            "mean": float(np.mean(bias_shifts)),
            "abs_mean": float(np.mean(np.abs(bias_shifts))),
            "std": float(np.std(bias_shifts)),
            "shifted_liberal": shifted_more_liberal,
            "shifted_conservative": shifted_more_conservative,
            "no_shift": no_shift
        },

        "diversity": {
            "mean": float(np.mean(diversities)),
            "std": float(np.std(diversities)),
            "min": float(np.min(diversities)),
            "max": float(np.max(diversities))
        }
    }

    return stats


def compare_methods_bias(
    corpus: NewsCorpus,
    behaviors_df: pd.DataFrame,
    news_id_to_index: Dict[str, int],
    recommendation_files: Dict[str, str],
    model_name: str = "valurank/distilroberta-bias",
    max_users_per_method: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare recommendation methods on political bias metrics

    Args:
        corpus: NewsCorpus
        behaviors_df: User behaviors
        news_id_to_index: ID mapping
        recommendation_files: Method name -> file path
        model_name: HuggingFace bias classifier
        max_users_per_method: Max users to evaluate

    Returns:
        Comparison DataFrame
    """
    analyzer = PoliticalBiasAnalyzer(corpus, model_name=model_name)

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

        stats = compute_bias_statistics(evaluations)

        comparison_results.append({
            "Method": method_name,
            "N": stats["num_users"],
            "Avg_Bias_Shift": f"{stats['bias_shift']['mean']:.3f}",
            "Abs_Bias_Shift": f"{stats['bias_shift']['abs_mean']:.3f}",
            "Diversity": f"{stats['diversity']['mean']:.3f}",
            "Diversity_Std": f"{stats['diversity']['std']:.3f}",
            "Shifted_Liberal": stats['bias_shift']['shifted_liberal'],
            "Shifted_Conservative": stats['bias_shift']['shifted_conservative'],
            "No_Shift": stats['bias_shift']['no_shift']
        })

    # Save detailed evaluations
    for method_name, evaluations in all_evaluations.items():
        filename = f"bias_eval_{method_name.lower().replace(' ', '_')}.json"
        eval_dicts = [
            {
                "user_id": e.user_id,
                "history_mean_bias": e.history_mean_bias,
                "recommended_mean_bias": e.recommended_mean_bias,
                "bias_shift": e.bias_shift,
                "bias_diversity": e.bias_diversity
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
    print("POLITICAL BIAS ANALYSIS")
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

    # Choose bias detection model
    # MODEL_NAME = "valurank/distilroberta-bias"
    MODEL_NAME = "valurank/distilroberta-bias"

    print(f"\nBias Model: {MODEL_NAME}")
    print("Note: First run downloads model (~250MB)")

    # Run comparison
    comparison_df = compare_methods_bias(
        corpus,
        behaviors,
        news_id_to_index,
        recommendation_files,
        model_name=MODEL_NAME,
        max_users_per_method=None
    )

    print("\nResults:")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv("bias_analysis.csv", index=False)
    print("Saved: bias_analysis.csv")

if __name__ == "__main__":
    main()
