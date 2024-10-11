from typing import Any, Optional
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, RobertaTokenizer, DistilBertTokenizer, T5Tokenizer
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from torch import cuda
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, DistilBertModel, T5ForConditionalGeneration
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from io import BytesIO
# from minio import Minio

data_path = "/scratch/jsb742/sampling/sampling_gpt/images/training_images/"
test_data_path = "/scratch/jsb742/sampling/sampling_gpt/images/validation_images/"
# Constants
MAX_LEN = 128
BATCH_SIZE = 16
IMAGE_DIR = data_path
NUM_EPOCHS = 15

class MultiModel:
    def __init__(self, model_name: Optional[str], training_data: Optional[pd.DataFrame], test_data: Optional[pd.DataFrame]):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_model_acc: Dict[str, str] = None
        self.training_data = training_data
        self.test_data = test_data
        self.run_clf = False
        self.base_model = None
        self.minio = False
        self.model = None

    def get_train_dataset(self, data):
        train_dataset = CustomDataset(
            dataframe=data,
            tokenizer=self.tokenizer,
            image_dir=data_path,
            max_len=MAX_LEN,
            image_bucket_name=None,
            transform=self.transform,
            minio=self.minio)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        return train_dataloader

    def get_validation_dataset(self):
        val_dataset = CustomDataset(
            dataframe=self.test_data,
            tokenizer=self.tokenizer,
            image_dir=test_data_path,
            max_len=MAX_LEN,
            image_bucket_name=None,
            transform=self.transform,
            minio=self.minio)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return val_dataloader

    def get_inference_dataset(self, dataset):
        inference_dataset = InferenceDataset(
            dataframe=dataset,
            tokenizer=self.tokenizer,
            image_dir=data_path,
            max_len=MAX_LEN,
            transform=self.transform,
            minio=self.minio
        )
    
        return DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False)

    def set_clf(self, set: bool):
        self.run_clf = set

    def get_clf(self):
        return self.run_clf

    def get_last_model_acc(self):
        return self.last_model_acc

    def set_train_data(self, train):
        self.training_data = train

    def get_train_data(self):
        return self.training_data
        
    def get_base_model(self):
        return self.base_model

    def train_data(self, df: pd.DataFrame):
        
        train_dataloader = self.get_train_dataset(df)
        val_dataloader = self.get_validation_dataset()
        
        self.model = MultiModalModel(num_labels=2)
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            total_loss = 0  # to keep track of training loss
        
            for batch in train_dataloader:
                
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
        
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, image=images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                loss.backward()
        
    
                optimizer.step()
                
        
            # Compute average training loss
            avg_train_loss = total_loss / len(train_dataloader)
        
            # Validation loop
            self.model.eval()
            val_preds, val_true = [], []
            val_total_loss = 0
        
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, image=images)
                    
                    loss = criterion(outputs, labels)
                    val_total_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
        
            eval_accuracy = accuracy_score(val_true, val_preds)
            eval_precision = precision_score(val_true, val_preds)
            eval_recall = recall_score(val_true, val_preds)
            eval_f1 = f1_score(val_true, val_preds)
            eval_loss = val_total_loss / len(val_dataloader)
            
            scheduler.step(eval_loss)
        
        
            # Print metrics
            print(f"Epoch: {epoch+1}/{NUM_EPOCHS}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {eval_loss:.4f}")
            print(f"Validation Accuracy: {eval_accuracy:.4f}\n")
            print(f"Validation Precision: {eval_precision:.4f}\n")
            print(f"Validation Recall: {eval_recall:.4f}\n")
            print(f"Validation F1-score: {eval_f1:.4f}\n")

        result = {
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "eval_precision": eval_precision,
                "eval_recall": eval_recall,
                "eval_f1": eval_f1
            }

        return result

    def get_inference(self, df: pd.DataFrame) -> torch.Tensor:
        dataloader = self.get_inference_dataset(df)
        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = []
    
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['image'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, image=images)
                
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
    
        return predictions

    def load_model(self, model_load_path):
        self.model = MultiModalModel(num_labels=2)
        if model_load_path:
            if self.device == 'cpu':
                self.model.load_state_dict(torch.load(model_load_path, map_location=self.device), strict=False)
            else:
                self.model.load_state_dict(torch.load(model_load_path), strict=False)
            

    def update_model(self, model_name, model_acc, save_model: bool):
        if save_model:
            torch.save(self.model.state_dict(), model_name)
        self.last_model_acc = {model_name: model_acc}
        self.load_model(model_name)
        self.base_model = model_name
        


class CrossAttention(nn.Module):
    def __init__(self, text_dim, image_dim):
        super(CrossAttention, self).__init__()
        
        # For text -> image
        self.query_text = nn.Linear(text_dim, text_dim)
        self.key_image = nn.Linear(image_dim, text_dim)
        self.value_image = nn.Linear(image_dim, image_dim)
        
        # For image -> text
        self.query_image = nn.Linear(image_dim, image_dim)
        self.key_text = nn.Linear(text_dim, image_dim)
        self.value_text = nn.Linear(text_dim, text_dim)

    def forward(self, text_features, image_features):
        # Text attending to Image
        attention_weights_text = torch.matmul(self.query_text(text_features), self.key_image(image_features).transpose(1, 2))
        attention_weights_text = nn.functional.softmax(attention_weights_text, dim=-1)
        attended_image = torch.matmul(attention_weights_text, self.value_image(image_features))

        # Image attending to Text
        attention_weights_image = torch.matmul(self.query_image(image_features), self.key_text(text_features).transpose(1, 2))
        attention_weights_image = nn.functional.softmax(attention_weights_image, dim=-1)
        attended_text = torch.matmul(attention_weights_image, self.value_text(text_features))

        return attended_text, attended_image

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, text_dim, image_dim, num_heads, hidden_dim, output_dim):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # These linear layers project the inputs to multiple heads
        self.text_query = nn.Linear(text_dim, hidden_dim, bias=False)
        self.text_key = nn.Linear(text_dim, hidden_dim, bias=False)
        self.text_value = nn.Linear(text_dim, hidden_dim, bias=False)

        self.image_query = nn.Linear(image_dim, hidden_dim, bias=False)
        self.image_key = nn.Linear(image_dim, hidden_dim, bias=False)
        self.image_value = nn.Linear(image_dim, hidden_dim, bias=False)

        # Final projection layer
        self.out_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, text_features, image_features):
        # print(text_features.shape)
        Q_text = self.text_query(text_features)
        K_text = self.text_key(text_features)
        V_text = self.text_value(text_features)

        Q_image = self.image_query(image_features)
        K_image = self.image_key(image_features)
        V_image = self.image_value(image_features)

        # Split the hidden dimension into num_heads
        Q_text = Q_text.view(Q_text.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_text = K_text.view(K_text.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_text = V_text.view(V_text.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q_image = Q_image.view(Q_image.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_image = K_image.view(K_image.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_image = V_image.view(V_image.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate the attention scores
        attn_scores_text_image = torch.matmul(Q_text, K_image.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_scores_image_text = torch.matmul(Q_image, K_text.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Normalize scores
        attn_probs_text_image = F.softmax(attn_scores_text_image, dim=-1)
        attn_probs_image_text = F.softmax(attn_scores_image_text, dim=-1)

        # Apply attention
        attn_output_text_image = torch.matmul(attn_probs_text_image, V_image)
        attn_output_image_text = torch.matmul(attn_probs_image_text, V_text)

        # Concatenate the results across the heads
        attn_output_text_image = attn_output_text_image.transpose(1, 2).contiguous().view(text_features.size(0), -1)
        attn_output_image_text = attn_output_image_text.transpose(1, 2).contiguous().view(image_features.size(0), -1)

        # Project to output dimension
        output_text_image = self.out_proj(attn_output_text_image)
        output_image_text = self.out_proj(attn_output_image_text)

        return output_text_image, output_image_text


class MultiModalModel(nn.Module):
        def __init__(self, num_labels):
            super(MultiModalModel, self).__init__()
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Load pre-trained models
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.resnet = models.efficientnet_v2_m(pretrained=True)

            # Remove the final classification layer of ResNet
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
            # self.mhca = MultiHeadCrossAttention(text_dim=768, image_dim=1280, num_heads=4, hidden_dim=512, output_dim=2048)
            self.mhca = MultiHeadCrossAttention(text_dim=768, image_dim=1280, num_heads=4, hidden_dim=512, output_dim=2048)

            self.classifier = nn.Sequential(
                nn.Linear(2816, 512), 
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_labels)
            )

        def forward(self, input_ids, attention_mask, image):
            # Forward pass through BERT
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_features = outputs['last_hidden_state'][:, 0, :]  # CLS token output as text feature
            # print(text_features.shape)

            # Forward pass through ResNet
            image_features = self.resnet(image)
            image_features = image_features.view(image_features.size(0), -1)  # Flatten the output
            # print(image_features.shape)
            # image_features = nn.Linear(1280, 2048).to(device)(image_features)

            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(1)
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(1)
                
            # print(text_features.shape, image_features.shape)
            attended_text, attended_image = self.mhca(text_features, image_features)

            attended_text = attended_text.squeeze(1)  # shape: [16, 768]
            attended_image = attended_image.squeeze(1) # shape: [16, 2048]
            # print(attended_text.shape, attended_image.shape)

            self.image_projection = torch.nn.Linear(2048, 768).to(self.device)
            attended_image = self.image_projection(attended_image)
            combined_features = torch.cat((attended_text, attended_image), dim=-1)

            logits = self.classifier(combined_features)

            return logits


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_dir, max_len, image_bucket_name, transform=None, minio= False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.max_len = max_len
        self.transform = transform
        self.minio = minio

    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['training_text']
        label = row['label']
        
        # Processing text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        # Processing images
        # img_path = os.path.join(self.image_dir, str(label), img_name + ".jpg")
        if self.minio:
            bucket, img_path = row["image_path"].split("/")
            image_data = client.get_object(bucket, img_path)
            image_bytes = image_data.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            img_path = self.image_dir+row['image_path']
            image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }


class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_dir, max_len, transform=None, minio=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.minio=minio
        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['title']
        img_path = row['image_path']
        
        # Processing text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        if self.minio:
            img_path = row["image_path"]
            image_data = client.get_object(bucket_name="images-november" ,object_name=img_path)
            image_bytes = image_data.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            img_path = self.image_dir+row['image_path']
            image = Image.open(img_path).convert("RGB")
            
        if self.transform:
            image = self.transform(image)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image
        }