## CBS approch
run: pip install -r requirements.txt

run: main_cluster.py -sample_size 200 -filename "shark_trophy" -balance False -sampling thompson -filter_label True -model_finetune "bert-base-uncased" -labeling "gpt" -model "text" -baseline 0.5 -metric "f1"

arguments:
sample_size 200 -> every iteration is select a sample size of 200
filename -> The csv file with the complite collection of data with a title (text) column for labeling
balance -> If you wanna balance the data with undersampling
sampling -> the samplinh method use. Can choose between thompson sampling, random sampling
filter_label-> If you wanna filter labels based on positive samples
model_finetune-> the model used for finetune during active learning
labeling -> where the labels are coming from: GPT, LLAMA, FILE
model -> choose between text only or multi-modal model
metric -> The type of metric to be used for baseline, f1, accuracy, recall, precision
baseline -> The initial baseline for the metric







