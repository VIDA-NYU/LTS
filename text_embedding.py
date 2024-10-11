import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
import umap


class BertTextEmbedder:
    def __init__(self, model_name='bert-base-uncased', save_embedding=True):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_embedding = save_embedding
        self.model.to(self.device)
        self.model.eval()

    def get_bert_embeddings(self, sentences):
        input_ids = []
        attention_masks = []

        for sentence in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                                sentence,
                                add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
                                max_length=64,             # Adjust sentence length
                                pad_to_max_length=True,    # Pad/truncate sentences
                                return_attention_mask=True,# Generate attention masks
                                return_tensors='pt',       # Return PyTorch tensors
                        )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        print(len(input_ids), len(attention_masks))
        results = self.generate_bert_embeddings( input_ids, attention_masks, 100)
        return results


    def generate_bert_embeddings(self, input_ids, attention_masks, batch_size):
        results = []
        model = self.model.eval()

        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                print(f"Getting embedding for batch {i}")
                batch_input_ids = torch.cat(input_ids[i:i+batch_size], dim=0)
                batch_attention_mask = torch.cat(attention_masks[i:i+batch_size], dim=0)
                outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
                batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
                if self.save_embedding:
                    results.append(batch_embeddings)
                else:
                    results.append(self.apply_umap_projection(batch_embeddings))

        results = np.concatenate(results, axis=0)
        print(f"Finished processing for {i} + {batch_size}")
        return results

    def apply_umap_projection(self, bert_embeddings, n_components=2):
        print("Creating umap projections")
        umap_model = umap.UMAP(n_components=n_components, metric='euclidean')
        umap_projection = umap_model.fit_transform(bert_embeddings)
        return umap_projection


