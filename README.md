# AIDMS

This project implements and evaluates three different recommendation algorithms for news articles, with a focus on measuring information diversity and echo chamber effects.

## Recommendation Algorithms

We implement three recommendation algorithms:

1. **Pure Relevance** (`recommendation_algorithms/pure_relevance/`)

   - Simple similarity-based ranking
   - Recommends articles most similar to user's reading history using cosine similarity
   - Creates user profile by averaging history embeddings

2. **Calibrated Diversity** (`recommendation_algorithms/calibrated_diversity/`)

   - Uses Maximal Marginal Relevance (MMR) algorithm
   - Balances relevance with diversity using a lambda parameter
   - Score = λ × relevance + (1-λ) × diversity

3. **Serendipity-Aware** (`recommendation_algorithms/serendipity/`)
   - Deliberately includes unexpected content from different topics/viewpoints
   - Mixes relevant articles with distant (but quality) content
   - Configurable serendipity ratio (default: 30% unexpected content)

## Evaluation Methodologies

We evaluate these recommendation algorithms using four different methodologies:

1. **Centroid Method (Embedding Drift)** (`evaluation/centroid_method.py`)

   - Computes embedding drift by comparing centroid vectors of user history vs recommendations
   - Measures how recommendations diverge from user's existing preferences in embedding space
   - High drift = broader exposure, diverse recommendations
   - Low drift = echo chamber tightening, similar recommendations
   - Visualizes actual centroid locations in 2D space using PCA

2. **LLM Evaluation** (`evaluation/llm_method.py`)

   - Uses large language models (GPT-2) to assess recommendation quality and diversity
   - Evaluates three key metrics:
     - **Novelty**: How new/different the recommended content is from user history (1-5 scale)
     - **Perspective Contrast**: How different the viewpoints are (1-5 scale)
     - **Framing Difference**: Whether recommendations use different political framing (%)
   - Provides detailed reasoning for each evaluation

3. **Diversity Method (Echo Chamber Analysis)** (`evaluation/diversity_method.py`)

   - Measures echo chamber effects and diversity in recommendations using semantic clustering
   - Computes pairwise distances between articles in embedding space
   - Key metrics:
     - **Echo Score**: Lower is better, measures how similar recommendations are to history
     - **History Diversity**: Average pairwise distance in user's reading history
     - **Recommendation Diversity**: Average pairwise distance in recommended articles
     - **Echo Chamber %**: Percentage of recommendations that reinforce existing views
     - **Bubble Breaking %**: Percentage of recommendations that introduce new perspectives
   - Echo Chamber Score = diversity_of_history / diversity_of_recommendations
     - Score > 1.0 = Echo chamber (recommendations more clustered than history)
     - Score < 1.0 = Bubble breaking (recommendations more diverse than history)

4. **Political Bias Classifier** (`evaluation/sentiment_method.py`)

   - Analyzes political bias/leaning of news articles using HuggingFace models
   - Uses `valurank/distilroberta-bias` model to classify articles on political spectrum
   - Measures:
     - Mean bias shift between history and recommendations
     - Bias diversity in recommendations
     - Tracks whether recommendations introduce diverse political perspectives

## Project Structure

```
AIDS/
├── recommendation_algorithms/
│   ├── pure_relevance/
│   ├── calibrated_diversity/
│   ├── serendipity/
│   ├── news_corpus.py          # Corpus and embedding creation
│   ├── demo.py                  # Demo with sample data
│   └── full_run.py              # Full pipeline on MIND dataset
├── evaluation/
│   ├── centroid_method.py       # Centroid-based embedding drift evaluation
│   ├── llm_method.py            # LLM-based evaluation (novelty, perspective, framing)
│   ├── diversity_method.py      # Echo chamber and diversity analysis
│   ├── sentiment_method.py      # Political bias classifier
│   └── visualize_results.py     # Generate visualizations for all evaluation results
├── recommended_results/         # Generated recommendation JSON files
├── eval_results/                # Evaluation results and comparisons
│   ├── echo_chamber_analysis.csv    # Diversity and echo chamber metrics
│   ├── llm_evaluation.csv           # LLM evaluation summary
│   ├── llm_reasoning/               # Detailed LLM evaluation JSON files
│   └── *.png                        # Visualization outputs
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo

Run a demo with sample data:

```bash
python recommendation_algorithms/demo.py
```

### Full Pipeline

Generate recommendations on the MIND dataset:

```bash
python recommendation_algorithms/full_run.py
```

### Evaluation

Run individual evaluation methods:

```bash
# Centroid method (embedding drift)
python evaluation/centroid_method.py

# LLM evaluation (novelty, perspective, framing)
python evaluation/llm_method.py

# Diversity method (echo chamber analysis)
python evaluation/diversity_method.py

# Political bias classifier
python evaluation/sentiment_method.py
```

Generate all visualizations:

```bash
python evaluation/visualize_results.py
```

This creates visualizations for:

- Centroid locations in embedding space (PCA-reduced)
- LLM evaluation box plots and distributions
- Diversity metrics and echo chamber analysis
- Combined comparison across all methods

## Results

Recommendation results are saved in `recommended_results/` and evaluation comparisons are saved in `eval_results/`.
