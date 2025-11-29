import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Optional

"""
Create embeddings for news corpus
Default model: 'all-MiniLM-L6-v2' (Sentence-BERT)
"""

class NewsCorpus:
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
