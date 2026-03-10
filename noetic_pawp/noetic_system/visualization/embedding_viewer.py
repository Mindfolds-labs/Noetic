from __future__ import annotations

import numpy as np


class EmbeddingViewer:
    """UMAP projection helper for concept inspection and trajectories."""

    def project(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("umap-learn is required for embedding projection") from exc
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)

    @staticmethod
    def nearest_neighbors(embeddings: np.ndarray, query_index: int, topk: int = 5) -> list[int]:
        q = embeddings[query_index]
        dists = np.linalg.norm(embeddings - q, axis=1)
        return np.argsort(dists)[:topk].tolist()
