# import pandas as pd
# from hdbscan import HDBSCAN

# class TextClusterer:
#     def __init__(self, min_cluster_size=5, min_samples=1, cluster_selection_epsilon=0.5):
#         self.clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
#                                   min_samples=min_samples,
#                                   cluster_selection_epsilon=cluster_selection_epsilon)

#     def fit_predict(self, embeddings):
#         cluster_labels = self.clusterer.fit_predict(embeddings)
#         return cluster_labels
