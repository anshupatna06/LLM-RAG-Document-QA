from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.

        Args:
            model_name (str): Name of the sentence-transformer model.
        """
        assert isinstance(model_name, str) and len(model_name.split()) == 1, \
            "model_name must be a valid HuggingFace model id"
        
        self.model = SentenceTransformer(model_name)

    def __call__(self, text):
        """
        Generates embedding for a single text input.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: Embedding vector.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding)
