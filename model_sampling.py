import numpy as np
import pandas as pd

class ModelSampler:
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits

        try:
            self.selected_ids = set(np.loadtxt('selected_ids.txt', dtype=str))
        except IOError:
            self.selected_ids = set()

    def get_sample_data(self, df, sample_size):

        if "predicted_label" in df.columns:
            df = df[df["predicted_label"] == 1]

        unique_clusters = df['label_cluster'].unique()
        samples_per_cluster = int(sample_size / self.n_bandits)

        sampled_data = pd.DataFrame()

        df = df[~df['id'].isin(self.selected_ids)]
        # Sample data from each cluster
        for cluster in unique_clusters:
            cluster_data = df[df['label_cluster'] == cluster]

            if len(cluster_data) >= samples_per_cluster:
                sampled_cluster_data = cluster_data.sample(n=samples_per_cluster, random_state=42)
            else:
                sampled_cluster_data = cluster_data

            # Concatenate the sampled cluster data to the sampled_data DataFrame
            sampled_data = pd.concat([sampled_data, sampled_cluster_data], ignore_index=True)

        # Add the IDs of sampled data to the selected_ids set
        self.selected_ids.update(sampled_cluster_data['id'])
        with open('selected_ids.txt', 'w') as f:
            f.write('\n'.join(self.selected_ids))

        return sampled_data, "random"
