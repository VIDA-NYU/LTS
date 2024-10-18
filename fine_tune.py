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
import openai
from openai import OpenAI
import os
import pandas as pd


class Labeling:
    def __init__(self, label_model= "llama"):
        self.label_model = label_model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def generate_prompt(self, title):
        if self.label_model == "llama":
            return self.generate_prompt_llama(title)
        elif self.label_model=="gpt":
            return self.generate_prompt_gpt(title)
        else:
            return None


    def generate_prompt_llama(self, title: str) -> str:
        return f"""### Instruction: {self.prompt_llama}
                ### Input:
                {title.strip()}
                """

    def generate_prompt_gpt(self, title):
        return f'''You are labeling tool to create labels for a classification task .
                             I will provide text data from an advertisement of a product.
                             The product should be classified in two labels:
                             Label 1: relevant animal - if the product is from any of those 3 animals: Shark, Ray or Chimaeras. It should be from a real animal. Not an image or plastic for example.
                             Label 2: not a relevant animal - if the product is from any other animal that is not Shark, Ray, or Chimaeras, or if the product is 100% synthetic (vegan).
                             For products such as teeth, if it's mentioned that is only one tooth, you can label it as not a relevant animal. I am only interested in more than one tooth. Also if mentions it's a fossil, we are not interested. you can label it as not a relevant animal.
                             Return only one of the two labels: relevant animal or not a relevant animal, no explanation is necessary.
                             Exemple:
                             1. Advertisement: Great White Shark Embroidered Patch Iron on Patch For Clothes
                             Label: not a relevant animal

                             The product in example 1 is a piece of clothing with an animal embroidered. The product is not MADE by any animal product.

                             2. Advertisement: (sj480-50) 6" White Tip Reef SHARK jaw love sharks jaws teeth Triaenodon obesus
                             Label: relevant animal

                             The product in example 2 is selling a shark jaw. 100% animal product in this case.

                             3. Advertisement: Wholesale Group - 20 Perfect 5/8" Modern Tiger Shark Teeth
                             Label: relevant animal

                             In example 3 we have a set of 20 teeth. In this case is True.

                             4. Advertisement: Mario Buccellati, a Rare and Exceptional Italian Silver Goat For Sale at 1stDibs
                             Label: not a relevant animal

                             This example 4 is also not an animal product. The goat in the ad is made out of silver and it's not the animal we are interested.

                             5. Advertisement: HUGE SHARK TOOTH FOSSIL 3&1/4" GREAT Serrations Upper Principal
                             Label: not a relevant animal

                             This is a product from a shark, but is not an animal product because it's only one tooth and it's a fossil.

                             6. Advertisement: {title}
                             Label:

                             '''

    def generate_llama_prompt(self):
        f'''You are labeling tool to create labels for a classification task .
                             I will provide text data from an advertisement of a product.
                             The product should be classified in two labels:
                             relevant animal - if the product is from any of those 3 animals: Shark, Ray or Chimaeras. It should be from a real animal. Not an image or plastic for example.
                             not a relevant animal - if the product is from any other animal that is not Shark, Ray, or Chimaeras, or if the product is 100% synthetic (vegan).
                             For products such as teeth, if it's mentioned that is only one tooth, you can label it as not a relevant animal. I am only interested in more than one tooth. Also if mentions it's a fossil, we are not interested. you can label it as not a relevant animal.
                             Return only one of the two labels, no explanation is necessary.
                             Exemple:
                             1. Advertisement: Great White Shark Embroidered Patch Iron on Patch For Clothes
                             Label: not a relevant animal

                             The product in example 1 is a piece of clothing with an animal embroidered. The product is not MADE by any animal product.

                             2. Advertisement: (sj480-50) 6" White Tip Reef SHARK jaw love sharks jaws teeth Triaenodon obesus
                             Label: relevant animal

                             The product in example 2 is selling a shark jaw. 100% animal product in this case.

                             3. Advertisement: Wholesale Group - 20 Perfect 5/8" Modern Tiger Shark Teeth
                             Label: relevant animal

                             In example 3 we have a set of 20 teeth. In this case is True.

                             4. Advertisement: Mario Buccellati, a Rare and Exceptional Italian Silver Goat For Sale at 1stDibs
                             Label: not a relevant animal

                             This example 4 is also not an animal product. The goat in the ad is made out of silver and it's not the animal we are interested.

                             5. Advertisement: HUGE SHARK TOOTH FOSSIL 3&1/4" GREAT Serrations Upper Principal
                             Label: not a relevant animal

                             This is a product from a shark, but is not an animal product because it's only one tooth and it's a fossil.
                             6. Advertisement:
                             '''

    def set_model(self):
        if self.label_model == "llama":
            checkpoint = "llama/"
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.prompt_llama = self.generate_llama_prompt()
            print("model Loaded")
        elif self.label_model == "gpt":
            self.model = OpenAI(api_key="xxxxxx")
        elif self.label_model =="file":
            self.model = None


    def predict_animal_product(self, row):
        label = Labeling.check_already_label(row) ## for test if you have data labeled
        if label:
            return label
        if self.label_model == "llama":
            return self.get_llama_label(row)
        elif self.label_model == "gpt":
            return self.get_gpt_label(row)
        elif self.label_model == "file":
            return self.get_file_label(row)
        else:
            raise ValueError("No model selected")


    def generate_inference_data(self, data, column):
        def truncate_string(s, max_length=2000):  # Adjust max_length as needed
            return s[:max_length] + '...' if len(s) > max_length else s

        if self.label_model != "file":
            examples = []
            for _, data_point in data.iterrows():
                examples.append(
                {
                    "id": data_point["id"],
                    "title": data_point["title"],
                    "training_text": data_point["clean_title"],
                    "text": self.generate_prompt(truncate_string(data_point[column])),
                }
                )
            data = pd.DataFrame(examples)
        return data



    def get_gpt_label(self, row):
        labels =  pd.read_csv("labaled_by_gpt.csv")
        id_ = row["id"]
        prompt = row["text"]
        if id_ in labels["id"].to_list():
            return labels.loc[labels["id"] == id_, "label"].values[0]
        else:
            response = self.model.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=100,
                    temperature=0.2,
                )
            return response.choices[0].message.content


    def get_llama_label(self, row):
        text = row["text"]
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
            results = self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
            try:
                answer = results.split("Response:\n")[2].split("\n")[0]
            except Exception:
                # Handle IndexError separately
                try:
                    answer = results.split("Response:\n")[1].split("\n")[0]
                except Exception:
                    # Handle any other exception
                    answer = 'not a relevant animal'
        return answer


    def get_file_label(self, row):
        raise NotImplementedError()

    @staticmethod
    def check_already_label(row):
        return None
        # labeled_data = pd.read_csv("all_labeled_data_gpt.csv")
        # if row["id"] in labeled_data["id"].values:
        #     # Retrieve the label for the corresponding id
        #     print("data already labeled")
        #     label = labeled_data.loc[labeled_data["id"] == row["id"], "label"].values[0]
        #     return label
        # else:
        #     return None
