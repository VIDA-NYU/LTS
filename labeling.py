import pandas as pd
import json
import re
from pprint import pprint

import torch
from datasets import Dataset, DatasetDict

from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class Labeling:
    def __init__(self, label_model= "llama"):
        self.label_model = label_model
        self.prompt = """
            Below is a advertisement title from a webcommerce. Asnwer if this advertisement contains any animal product.
            Such as animal body parts, or a live animal.
            The product should not just mention the animal, but the product itself must be made by an animal. Answer with "animal product" or "not animal product".
            """.strip()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def generate_prompt(self, title: str) -> str:
        return f"""### Instruction: {self.prompt}
                ### Input:
                {title.strip()}
                """

    def set_model(self):
        if self.label_model == "llama":
            self.model = AutoModelForCausalLM.from_pretrained("/scratch/jsb742/model").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("/scratch/jsb742/model")
            print("model Loaded")
        elif self.label_model == "gpt":
            raise NotImplementedError()
        else:
            raise NotImplementedError()


    def predict_animal_product(self, row):
        text = row["text"]
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
            results = self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
            try:
                answer = results.split("Response:\n")[2].split("\n")[0]
            except Exception:
                answer = 'not animal product'
        return answer

    def generate_inference_data(self, data, column):
        examples = []
        for _, data_point in data.iterrows():
            examples.append(
            {
                "id": data_point["id"],
                "title": data_point["title"],
                "training_text": data_point["clean_text"],
                "text": self.generate_prompt(data_point[column]),
            }
            )
        test_df = pd.DataFrame(examples)
        return test_df
