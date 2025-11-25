import pandas as pd
from recommendations import NewsCorpus, PureRelevanceRecommender, CalibratedDiversityRecommender, SerendipityAwareRecommender

"""
Demo with sample data
"""
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

print(" Pure Relevance")
print(" Calibrated Diversity (λ=0.7)")
print(" Serendipity-Aware (30% unexpected)")

# Simulate user with history
print("\n4. Simulating user with reading history...")
# Read some political articles
user_history = [0, 1, 5, 8]
print("User has read articles:", user_history)

# get recommendations from each algorithm
print("\n5. Getting recommendations (k=10)...")

recs1 = rec1.recommend(user_history, k=10)
recs2 = rec2.recommend(user_history, k=10)
recs3 = rec3.recommend(user_history, k=10)

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n Pure Relevance:")
for i, idx in enumerate(recs1[:5], 1):
    print(f"  {i}. [{idx}] {corpus.df.iloc[idx]['title']}")

print("\n Calibrated Diversity (MMR):")
for i, idx in enumerate(recs2[:5], 1):
    print(f"  {i}. [{idx}] {corpus.df.iloc[idx]['title']}")

print("\n Serendipity-Aware:")
for i, idx in enumerate(recs3[:5], 1):
    print(f"  {i}. [{idx}] {corpus.df.iloc[idx]['title']}")

print("\n" + "="*70)
print("Demo complete!")
