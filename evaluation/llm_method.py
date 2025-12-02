"""
LLM-based Evaluation of News Recommendations
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
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
class LLMEvaluation:
    user_id: str
    novelty_score: float
    perspective_contrast_score: float
    political_framing_different: int


class LLMEvaluator:
    """
    LLM evaluator
    """

    def __init__(
        self,
        corpus: NewsCorpus,
        model: str = "microsoft/phi-2",
        use_gpu: bool = True,
        batch_size: int = 1
    ):
        """
        Args:
            corpus: NewsCorpus with article data
            model: HuggingFace model name
            use_gpu: Whether to use GPU if available
            batch_size: Number of evaluations to batch (1 is safest)
        """
        self.corpus = corpus
        self.model_name = model
        self.df = corpus.df
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.llm_pipeline = None

    def _load_model(self):
        """Load HuggingFace model"""
        if self.llm_pipeline is not None:
            return

        try:
            from transformers import pipeline
            import torch

            print(f"Loading model: {self.model_name}")

            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                device = 0
                dtype = torch.float16
            else:
                device = -1
                dtype = torch.float32

            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=device,
                torch_dtype=dtype,
                max_length=2048,
                trust_remote_code=True  # Required for some models like Phi-2
            )

            # Set pad token if needed
            if self.llm_pipeline.tokenizer.pad_token is None:
                self.llm_pipeline.tokenizer.pad_token = self.llm_pipeline.tokenizer.eos_token

            print("Model loaded successfully")

        except ImportError as e:
            print("ERROR: pip install transformers torch accelerate")
            raise e

    def _create_prompt(self, history_articles: List[Dict], recommended_articles: List[Dict]) -> str:
        history_sample = history_articles[:8]
        rec_sample = recommended_articles[:8]

        history_text = "USER HISTORY:\n"
        for i, art in enumerate(history_sample, 1):
            history_text += f"{i}. [{art['topic']}] {art['title']}\n"

        rec_text = "\nRECOMMENDATIONS:\n"
        for i, art in enumerate(rec_sample, 1):
            rec_text += f"{i}. [{art['topic']}] {art['title']}\n"

        prompt = f"""You are evaluating news recommendations. Given a user's reading history and new recommendations, rate the recommendations on 3 metrics.

{history_text}
{rec_text}

Rate these recommendations:
- novelty: How different are the topics? 1=same topics as history, 5=very different topics
- perspective: How diverse are the viewpoints? 1=same viewpoint as history, 5=diverse viewpoints
- framing: Does the political framing differ? 0=similar framing, 1=different political framing

Respond ONLY with valid JSON in this exact format:
{{"novelty": <number 1-5>, "perspective": <number 1-5>, "framing": <0 or 1>}}

JSON response:"""

        return prompt

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response (just scores)"""
        try:
            response = response.strip()

            # Remove markdown
            if response.startswith("```"):
                response = response[response.find('\n')+1:]
            if response.endswith("```"):
                response = response[:response.rfind("```")]
            response = response.strip()

            # Find JSON
            start = response.find('{')
            end = response.rfind('}') + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)

                return {
                    "novelty": float(parsed.get("novelty", 3)),
                    "perspective": float(parsed.get("perspective", 3)),
                    "framing": int(parsed.get("framing", 0))
                }
            else:
                raise ValueError("No JSON found")

        except Exception as e:
            # Return neutral defaults on parse error
            return {"novelty": 3.0, "perspective": 3.0, "framing": 0}

    def _call_llm(self, prompt: str) -> str:
        """Call HuggingFace LLM"""
        self._load_model()

        if self.llm_pipeline is None:
            raise RuntimeError("Model failed to load properly")

        try:
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=100,
                temperature=0.1,  # Low temp for consistency
                do_sample=True,
                top_p=0.9,
                return_full_text=False,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
            )

            return response[0]['generated_text']

        except Exception as e:
            print(f"ERROR calling LLM: {e}")
            # Return empty string to trigger default values
            return ""

    def evaluate_single_user(
        self,
        user_id: str,
        history_indices: List[int],
        recommended_indices: List[int]
    ) -> LLMEvaluation:
        """Evaluate recommendations for a single user"""

        # Get article data
        history_articles = [
            {
                'title': self.df.iloc[idx]['title'],
                'topic': self.df.iloc[idx]['topic']
            }
            for idx in history_indices
        ]

        recommended_articles = [
            {
                'title': self.df.iloc[idx]['title'],
                'topic': self.df.iloc[idx]['topic']
            }
            for idx in recommended_indices
        ]

        # Create prompt
        prompt = self._create_prompt(history_articles, recommended_articles)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse
        parsed = self._parse_response(response)

        return LLMEvaluation(
            user_id=user_id,
            novelty_score=parsed["novelty"],
            perspective_contrast_score=parsed["perspective"],
            political_framing_different=parsed["framing"]
        )

    def evaluate_recommendations_file(
        self,
        recommendations_file: str,
        behaviors_df: pd.DataFrame,
        news_id_to_index: Dict[str, int],
        max_users: Optional[int] = None
    ) -> List[LLMEvaluation]:
        """
        Evaluate all recommendations in a file
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

        # Filter to users with history
        valid_recs = [
            rec for rec in recommendations
            if rec["user_id"] in user_history_map
        ]

        print(f"Evaluating {len(valid_recs)} users...")

        # Evaluate with progress bar
        evaluations = []

        for rec in tqdm(valid_recs, desc="Evaluating"):
            user_id = rec["user_id"]
            recommended_indices = rec["recommended_indices"]
            history_indices = user_history_map[user_id]

            try:
                evaluation = self.evaluate_single_user(
                    user_id,
                    history_indices,
                    recommended_indices
                )
                evaluations.append(evaluation)

            except Exception as e:
                print(f"\nError on user {user_id}: {e}")
                continue

        return evaluations


def compute_statistics(evaluations: List[LLMEvaluation]) -> Dict:
    """Compute statistics from LLM evaluations"""

    novelty = [e.novelty_score for e in evaluations]
    perspective = [e.perspective_contrast_score for e in evaluations]
    framing = [e.political_framing_different for e in evaluations]

    return {
        "num_evaluations": len(evaluations),
        "novelty": {
            "mean": float(np.mean(novelty)),
            "median": float(np.median(novelty)),
            "std": float(np.std(novelty))
        },
        "perspective": {
            "mean": float(np.mean(perspective)),
            "median": float(np.median(perspective)),
            "std": float(np.std(perspective))
        },
        "framing": {
            "percent_different": float(np.mean(framing) * 100),
            "count": int(sum(framing))
        }
    }


def compare_all_methods(
    corpus: NewsCorpus,
    behaviors_df: pd.DataFrame,
    news_id_to_index: Dict[str, int],
    recommendation_files: Dict[str, str],
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_users_per_method: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare all recommendation methods using LLM evaluation
    """

    evaluator = LLMEvaluator(corpus, model=model_name)

    comparison_results = []
    all_evaluations = {}

    for method_name, file_path in recommendation_files.items():
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"{'='*60}")

        evaluations = evaluator.evaluate_recommendations_file(
            file_path,
            behaviors_df,
            news_id_to_index,
            max_users=max_users_per_method
        )

        all_evaluations[method_name] = evaluations

        stats = compute_statistics(evaluations)

        comparison_results.append({
            "Method": method_name,
            "N": stats["num_evaluations"],
            "Novelty": f"{stats['novelty']['mean']:.2f}",
            "Novelty_Std": f"{stats['novelty']['std']:.2f}",
            "Perspective": f"{stats['perspective']['mean']:.2f}",
            "Perspective_Std": f"{stats['perspective']['std']:.2f}",
            "Framing_Diff_%": f"{stats['framing']['percent_different']:.1f}%"
        })

        print(f"\n Novelty:      {stats['novelty']['mean']:.2f} ± {stats['novelty']['std']:.2f}")
        print(f" Perspective:  {stats['perspective']['mean']:.2f} ± {stats['perspective']['std']:.2f}")
        print(f" Diff Framing: {stats['framing']['percent_different']:.1f}%")

    # Save detailed results
    for method_name, evaluations in all_evaluations.items():
        filename = f"llm_eval_{method_name.lower().replace(' ', '_')}.json"
        eval_dicts = [
            {
                "user_id": e.user_id,
                "novelty": e.novelty_score,
                "perspective": e.perspective_contrast_score,
                "framing_different": e.political_framing_different
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
    print("LLM EVALUATION")
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

    # Choose model (instruction-tuned models work much better)
    MODEL_NAME = "microsoft/phi-2"  # Truly open, no authentication needed, good at following instructions
    # MODEL_NAME = "google/flan-t5-large"  # Alternative, good at following instructions
    # MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Another free option

    comparison_df = compare_all_methods(
        corpus,
        behaviors,
        news_id_to_index,
        recommendation_files,
        model_name=MODEL_NAME,
        max_users_per_method=None  # Test on 40 users first
    )

    comparison_df.to_csv("llm_evaluation.csv", index=False)
    print("\n Saved: llm_evaluation.csv")

if __name__ == "__main__":
    main()
