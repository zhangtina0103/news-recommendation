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

We evaluate these recommendation algorithms using three different methodologies:

1. **Centroid Method** (`evaluation/centroid_method.py`)

   - Computes embedding drift by comparing centroid vectors of user history vs recommendations
   - Measures how recommendations diverge from user's existing preferences
   - High drift = broader exposure, diverse recommendations
   - Low drift = echo chamber tightening, similar recommendations

2. **LLM Evaluation**

   - Uses large language models to assess recommendation quality and diversity
   - Evaluates semantic diversity and information breadth of recommendations

3. **Sentiment Analysis**
   - Analyzes sentiment distribution in recommended articles
   - Measures whether recommendations introduce diverse perspectives
   - Tracks sentiment shifts between user history and recommendations

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
│   └── centroid_method.py       # Centroid-based embedding drift evaluation
├── recommended_results/         # Generated recommendation JSON files
├── eval_results/                # Evaluation results and comparisons
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

Run the centroid method evaluation:

```bash
python evaluation/centroid_method.py
```

## Results

Recommendation results are saved in `recommended_results/` and evaluation comparisons are saved in `eval_results/`.
