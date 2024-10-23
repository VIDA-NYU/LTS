# Sampling Based Methods for Inner Product Sketching

This is the code for the paper "A Cost-Effective LLM-based Approach to Identify Wildlife
Trafficking in Online Marketplaces" submitted to VLDB 2024.


## Contents

This README file is divided into the following sections:

* [1. Requirements](#-1-requirements)
* [2. Setup ](#-2-setup)
* [3. Use-cases ](#-3-use-cases)



## 🚀 1. Requirements
The paper experiments were run using `Python 3.12.7` with the following required packages. They are also listed in the `requirements.txt` file.
- datasets==2.11.0
- gensim==4.3.2
- nltk==3.8.1
- numpy==1.24.2
- openai==1.51.2
- pandas==2.0.0
- peft==0.10.0
- Pillow==10.4.0
- scikit_learn==1.2.2
- scipy==1.14.1
- torch==2.2.0
- torchvision==0.17.0
- transformers==4.39.0

## 🔥 2 SETUP
### 2.1 Create a virtual environment (optional, but recommended)

To isolate dependencies and avoid library conflicts with your local environment, you may want to use a Python virtual environment manager. To do so, you should run the following commands to create and activate the virtual environment:
```bash
python -m venv ./venv
source ./venv/bin/activate
```

### 🔥 2.2 Make sure you have the required packages installed

You can install the dependencies using `pip`:
```
pip install -r requirements.txt
```
### 🔥 2.3 Lalebing options

- You will need to set the open AI key to use gpt4 on labeling.py -labeling parameter gpt.

- To use LLAMA you can add the path to LLAMA model on labeling.py with -labeling parameter llama.

- For a pre-labeled data you can set -labeling parameter to "file"

## 🔥 3 Following is how to reproduce the experiments needed for each use-case in the paper.
- Parameters:
- sample_size -> every iteration is select a sample size
- filename -> The csv file with the complite collection of data with a title (text) column for labeling
- val_path -> path to the validation data
- balance -> If you wanna balance the data with undersampling
- sampling -> the sampling method use. Can choose between thompson sampling, random sampling
- filter_label-> If you wanna filter labels based on positive samples
- model_finetune-> the model used for finetune in the first iteration
- labeling -> where the labels are coming from: GPT, LLAMA, FILE
- model -> choose between text only or multi-modal model
- metric -> The type of metric to be used for baseline, f1, accuracy, recall, precision
- baseline -> The initial baseline score for the metric
- cluster_size -> The size of the cluster

#### 3.1 leather Products

run: python main_cluster.py -sample_size 200 -filename "data_use_cases/data_leather" -val_path "data/leather_validation.csv" -balance False -sampling "gpt" -filter_label True -model_finetune "bert-base-uncased" -labeling "gpt" -model "text" -baseline 0.5 -metric "f1" -cluster_size "10"

#### 3.2 SHARKS
run: python main_cluster.py -sample_size 200 -filename "data_use_cases/shark_trophy" -val_path "data_use_cases/validation_sharks.csv"  -balance True -sampling thompson -filter_label True -model_finetune "bert-base-uncased" -labeling "gpt" -m -model "text" -baseline 0.5 -metric "f1 -cluster_size "5"

