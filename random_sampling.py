from typing import Any
import numpy as np
import os
import pandas as pd

class RandomSampler:
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits

        try:
            self.selected_ids = set(np.loadtxt('selected_ids.txt', dtype=str))
        except IOError:
            self.selected_ids = set()

    def get_sample_data(self, df, sample_size, filter_label: bool, trainer: Any):
        def get_sample(data, size):
            if data.empty:
                return pd.DataFrame()
            else:
                return data.sample(min(size, len(data)), random_state=42)


        unique_clusters = df['label_cluster'].unique()

        samples_per_cluster = int(sample_size / self.n_bandits)

        sampled_data = []

        df = df[~df['id'].isin(self.selected_ids)]
        if filter_label:
            if trainer.get_clf():
                df["predicted_label"] = trainer.get_inference(df)

        # Sample data from each cluster
        for cluster in unique_clusters:
            cluster_data = df[df['label_cluster'] == cluster]

            if filter_label:
                if "predicted_label" in cluster_data.columns:
                    pos = cluster_data[cluster_data["predicted_label"] == 1]
                    neg = cluster_data[cluster_data["predicted_label"] == 0]
                    n_sample = int(samples_per_cluster/2)

                    pos_cluster_data = get_sample(pos, n_sample)
                    neg_cluster_data = get_sample(neg, samples_per_cluster-len(pos_cluster_data))

                    sampled_data.append(pd.concat([pos_cluster_data, neg_cluster_data]).sample(frac=1))
                else:
                    sampled_data.append(get_sample(cluster_data, size=samples_per_cluster))
            else:
                sampled_data.append(get_sample(cluster_data, size=samples_per_cluster))


        sampled_data = pd.concat(sampled_data, ignore_index=True)

        # Add the IDs of sampled data to the selected_ids set
        self.selected_ids.update(sampled_data['id'])
        with open('selected_ids.txt', 'w') as f:
            f.write('\n'.join(self.selected_ids))

        return sampled_data, "random"

