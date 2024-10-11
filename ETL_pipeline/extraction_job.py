from typing import Any, Optional
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd


class ExtractionJob():
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained("numind/NuExtract", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract", trust_remote_code=True)

        self.schema = """{
            "animal_name": "",
            "animal_scientific_name": "",
            "animal_body_part": "",
            "product_type": "",
            "product_material": "",
            "product_brand": ""
        }"""


    def perform_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        # print(torch.cuda.is_available())  # Should return True if CUDA is available
        # print(torch.cuda.current_device())  # Should print the current device ID
        # print(torch.cuda.get_device_name(0))

        self.model.to(self.device)
        self.model.eval()
        df["text"] = df["title"] #+ " " + df["description"]
        results = []
        for _, row in df.iterrows():
            try:
                res = self.predict_NuExtract(row["text"], example=["","",""])
                print(res)
                results.append(res)
            except Exception as e:
                print(f"extraction error: {e}")
                results.append(self.schema)

        jsons = []
        for i in results:
            try:
                jsons.append(json.loads(i))
            except Exception as e:
                jsons.append(json.loads(self.schema))

        new_info = pd.json_normalize(jsons)
        new_info.columns = new_info.columns.str.lower()
        columns = []
        for column in new_info.columns:
            if column in ["animal_name",
                            "animal_scientific_name",
                            "animal_body_part",
                            "product_type",
                            "product_material",
                            "product_brand"]:
                new_info[column] = new_info[column].str.lower()
                columns.append(column)
        if columns:
            new_info = new_info[columns]
            new_info = self.fix_df(new_info)
            df = pd.concat([df, new_info], axis=1)
            df = df.loc[:,~df.columns.duplicated()]

        return df


    def fix_df(self, df: pd.DataFrame) -> pd.DataFrame:
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # If column is a list, convert it to a string
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
            # Otherwise, ensure it's a string
            else:
                df[col] = df[col].astype(str)

        # Fill NaN values with empty strings for object columns
        df = df.fillna('')

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        return df

    def predict_NuExtract(self, text, example=["", "", ""]):
        schema = json.dumps(json.loads(self.schema), indent=4)
        input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
        for i in example:
            if i != "":
                input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"

        input_llm +=  "### Text:\n"+text +"\n<|output|>\n"
        input_ids = self.tokenizer(input_llm, return_tensors="pt",truncation = True, max_length=4000).to(self.device)

        output = self.tokenizer.decode(self.model.generate(**input_ids)[0], skip_special_tokens=True)
        return output.split("<|output|>")[1].split("<|end-output|>")[0]



