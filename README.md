# 🐾 Wildlife Trafficking Detection

This repository contains the code used in the paper:

> **"A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces"**
> Accepted at **SIGMOD 2025**

---

## 📚 Table of Contents

1. [Requirements](#-requirements)
2. [Setup](#-setup)
3. [Use Cases & Reproduction](#-use-cases--reproduction)

---

## 📦 Requirements

Experiments were conducted using **Python 3.12.7**. All required dependencies are listed in `requirements.txt` and can be installed via pip.


---

## ⚙️ Setup

### 1. Create a Virtual Environment (Recommended)

Use a virtual environment to avoid dependency conflicts:

```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate     # For Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

🧪 Use Cases & Reproduction
To reproduce experiments from the paper, run main_cluster.py with the appropriate flags:

🔧 Required Parameters:

- -sample_size: Number of samples per iteration.

- -filename: Path to the dataset CSV file. Must contain a text column named 'title'.

- -val_path: Path to the validation dataset.

- -balance: Whether to balance the dataset via undersampling (bool).

- -sampling: Sampling strategy (string: "thompson" or "random").

- -filter_label: Whether to filter out negative samples. (bool)

- -model_finetune: Model name for fine-tuning in the first iteration (string: e.g., "bert-base-uncased").

- -labeling: Source of labels (string: gpt, llama, or file).

- -model: Choose model type (string: text, multi-modal).

- -metric: Evaluation metric used to compare models between iterations (string: "f1", "accuracy", "recall", "precision").

- -baseline: Initial baseline metric score for first iteration.

- -cluster_size: Number of clusters to use.



👜 Use Case 1: Leather Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/data_leather" \
  -val_path "data/leather_validation.csv" \
  -balance False \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 10
```

🦈 Use Case 2: Shark Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/shark_trophy" \
  -val_path "data_use_cases/validation_sharks.csv" \
  -balance True \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 5
```

📫 Contact
For questions or feedback, please open an issue or reach out via the contact information provided in the paper.

