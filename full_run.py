import os
import json
import numpy as np
import pandas as pd

from recommendations import (
    NewsCorpus,
    PureRelevanceRecommender,
    CalibratedDiversityRecommender,
    SerendipityAwareRecommender
)

MIND_PATH = "/content/MINDsmall_train"

# =============================================================
# LOAD MIND DATASET AND CONVERT TO NewsCorpus FORMAT
# =============================================================

def load_mind_as_articles(path=MIND_PATH):
    behaviors = pd.read_csv(
        os.path.join(path, "behaviors.tsv"),
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"]
    )

    news = pd.read_csv(
        os.path.join(path, "news.tsv"),
        sep="\t",
        header=None,
        names=["news_id", "category", "subcategory",
               "title", "abstract", "url",
               "title_entities", "abstract_entities"]
    )

    articles = pd.DataFrame({
        "title": news["title"].fillna(""),
        "content": news["abstract"].fillna(""),
        "topic": news["category"],
        "source": None,
        "political_leaning": None
    })

    news_id_to_index = {nid: idx for idx, nid in enumerate(news["news_id"])}
    return articles, behaviors, news_id_to_index


# =============================================================
# MAIN PIPELINE
# =============================================================

def main():
    print("Loading MIND-small from:", MIND_PATH)

    articles_df, behaviors, news_id_to_index = load_mind_as_articles()

    corpus = NewsCorpus(articles_df)
    corpus.create_embeddings()

    rel = PureRelevanceRecommender(corpus)
    div = CalibratedDiversityRecommender(corpus, lambda_param=0.7)
    ser = SerendipityAwareRecommender(corpus, serendipity_ratio=0.3)

    results_rel = []
    results_div = []
    results_ser = []

    print("Generating recommendations for first 10,000 users...")

    for idx, row in behaviors.head(600).iterrows():

        if pd.isna(row["history"]):
            continue

        history_ids = row["history"].split()

        user_history_indices = [
            news_id_to_index[nid]
            for nid in history_ids
            if nid in news_id_to_index
        ]

        if not user_history_indices:
            continue

        rec1 = rel.recommend(user_history_indices, k=10)
        rec2 = div.recommend(user_history_indices, k=10)
        rec3 = ser.recommend(user_history_indices, k=10)

        results_rel.append({
            "user_id": str(row["user_id"]),                     # ensure string
            "recommended_indices": [int(x) for x in rec1]       # convert numpy int64 → int
        })
        results_div.append({
            "user_id": str(row["user_id"]),
            "recommended_indices": [int(x) for x in rec2]
        })
        results_ser.append({
            "user_id": str(row["user_id"]),
            "recommended_indices": [int(x) for x in rec3]
        })


        if idx % 200 == 0:
            print(f"Processed {idx} users...")

    json.dump(results_rel, open("pure_relevance.json", "w"), indent=4)
    json.dump(results_div, open("calibrated_diversity.json", "w"), indent=4)
    json.dump(results_ser, open("serendipity.json", "w"), indent=4)

    print("Done! JSON files saved in current folder.")


if __name__ == "__main__":
    main()
