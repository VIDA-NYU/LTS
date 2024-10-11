import argparse
import pandas as pd
import numpy as np
from labeling import Labeling
from random_sampling import RandomSampler
from multimodel import MultiModel
from preprocessing import TextPreprocessor
from fine_tune import BertFineTuner
from thompson_sampling import ThompsonSampler
import nltk
import json

nltk.download('punkt')

import os
from LDA import LDATopicModel

def main():
    parser = argparse.ArgumentParser(prog="Sampling fine-tuning", description='Perform Sampling and fine tune')
    # parser.add_argument('-cluster', type=str, required=False,
    #                     help="Name of cluster type")
    parser.add_argument('-sampling', type=str, required=False,
                        help="Name of sampling method")
    parser.add_argument('-sample_size', type=int, required=False,
                        help="sample size")
    parser.add_argument('-filter_label', type=bool, required=False,
                        help="use model clf results to filter data")
    parser.add_argument('-balance', type=bool, required=False,
                        help="balance positive and neg sample")
    parser.add_argument('-model_finetune', type=str, required=False,
                        help="model base for fine tune")
    parser.add_argument('-labeling', type=str, required=False,
                        help="Model to be used for labeling or file if label already on file")
    parser.add_argument('-baseline', type=float, required=False,
                        help="The initial baseline metric")
    parser.add_argument('-filename', type=str, required=False,
                        help="The initial file to be used")
    parser.add_argument('-model', type=str, required=False,
                        help="The type of model to be finetune")
    parser.add_argument('-metric', type=str, required=False,
                        help="The type of metric to be used for baseline")


    args = parser.parse_args()

    # cluster = args.cluster
    sampling = args.sampling
    sample_size = args.sample_size
    filter_label = args.filter_label
    balance = args.balance
    model_finetune = args.model_finetune
    labeling = args.labeling
    baseline = args.baseline
    filename = args.filename
    model = args.model
    metric = args.metric


    preprocessor = TextPreprocessor()


    validation = pd.read_csv("validation_sharks_df.csv")
    validation = preprocessor.preprocess_df(validation)
    validation["training_text"] = validation["clean_title"]
    validation.to_csv("validation_sharks_df.csv", index=False)

    try:
        data = pd.read_csv(filename+"_lda.csv")
        n_cluster = data['label_cluster'].value_counts().count()
        print("using data saved on disk")
    except Exception:
        print("Creating LDA")
        data = pd.read_csv(filename+".csv")
        data = preprocessor.preprocess_df(data)
        lda_topic_model = LDATopicModel(num_topics=3)
        topics = lda_topic_model.fit_transform(data['clean_title'].to_list())
        data["label_cluster"] = topics
        n_cluster = data['label_cluster'].value_counts().count()
        print(n_cluster)
        data.to_csv(filename + "_lda.csv", index=False)
        print("LDA created")


    baseline = baseline

    if model == "text":
        trainer = BertFineTuner(model_finetune, None, validation)
    else:
        trainer = MultiModel(None, None, validation)

    labeler = Labeling(label_model=labeling)
    labeler.set_model()

    if sampling == "thompson":
        ## thompson sampler
        sampler = ThompsonSampler(n_cluster)
    elif sampling == "random":
        sampler = RandomSampler(n_cluster)
    else:
        raise ValueError("Choose one of thompson or random")


    labeler = Labeling(label_model=labeling)
    labeler.set_model()

    for i in range(10):
        sample_data, chosen_bandit = sampler.get_sample_data(data, sample_size, filter_label, trainer)
        ## Generate labels
        if labeling != "file":
            df = labeler.generate_inference_data(sample_data, 'clean_title')
            print("df for inference created")
            df["answer"] = df.apply(lambda x: labeler.predict_animal_product(x), axis=1)
            df["answer"] = df["answer"].str.strip()
            df["label"] = np.where(df["answer"] == 'relevant animal', 1, 0)
            if os.path.exists(f"data_labeled_{filename}.csv"):
                train_data = pd.read_csv(f"data_labeled_{filename}.csv")
                train_data = pd.concat([train_data, df])
                train_data.to_csv(f"data_labeled_{filename}.csv", index=False)
            else:
                df.to_csv(f"data_labeled_{filename}.csv", index=False)
        else:
            df = sample_data
        print(df["label"].value_counts())
        print(df["answer"].value_counts())

        # ADD POSITIVE DATA IF AVAILABLE

        if os.path.exists('positive_data.csv'):
            pos = pd.read_csv('positive_data.csv')
            df = pd.concat([df, pos]).sample(frac=1)
            print(f"adding positive data: {df['label'].value_counts()}")
        if balance:
            if len(df[df["label"]==1]) > 0:
                unbalanced = len(df[df["label"]==0]) / len(df[df["label"]==1]) > 2
                if unbalanced:
                    label_counts = df["label"].value_counts()
                    # Determine the number of samples to keep for each label
                    min_count = min(label_counts)
                    balanced_df = pd.concat([
                        df[df["label"] == 0].sample(min_count*2),
                        df[df["label"] == 1].sample(min_count)
                    ])

                    # Shuffle the rows
                    df = balanced_df.sample(frac=1).reset_index(drop=True)
                    print(f"Balanced data: {df.label.value_counts()}")
            else:
                # if i == 0: # if this is the first model training
                print("continue")
                continue

        ## FINE TUNE MODEL

        #previous model
        model_name = trainer.get_base_model()
        print(f"using model {model_name}")
        model_results = trainer.get_last_model_acc()
        if model_results:
            baseline = model_results[model_name]
            print(f"previous model {metric} metric baseline of: {baseline}")
        else:
            print(f"Starting with metric {metric} baseline {baseline}")
        print(f"Starting training")

        try:
            still_unbalenced = len(df[df["label"]==0]) / len(df[df["label"]==1])  >= 2
        except Exception:
            still_unbalenced = True
        print(f"Unbalanced? {still_unbalenced}")

        results, huggingface_trainer = trainer.train_data(df, still_unbalenced)
        reward_difference = results[f"eval_{metric}"] - baseline
        if reward_difference > 0:
            print(f"Model improved with {reward_difference}")
            model_name = f"models/fine_tunned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{metric}"], save_model=True)
            # df.to_csv("llama_training_data.csv", index=False)
            if os.path.exists(f'training_data_{filename}.csv'):
                train_data = pd.read_csv(f'training_data_{filename}.csv')
                df = pd.concat([train_data, df])
            df.to_csv(f'training_data_{filename}.csv', index=False)
            if os.path.exists('positive_data.csv'):
                os.remove('positive_data.csv')
            if filter_label:
                trainer.set_clf(True)
        else:
            #back to initial model
            trainer.update_model(model_name, baseline, save_model=False)
            # save positive sample
            if os.path.exists('positive_data.csv'):
                positive = pd.read_csv("positive_data.csv")
                df = df[df["label"]==1]
                df = pd.concat([df, positive])
                df = df.drop_duplicates()
            df[df["label"]==1].to_csv("positive_data.csv", index=False)


        if os.path.exists(f'model_results_{filename}.json'):
            with open(f'model_results_{filename}.json', 'r') as file:
                existing_results = json.load(file)
        else:
            existing_results = {}

        if existing_results.get(str(chosen_bandit)):
            existing_results[str(chosen_bandit)].append(results)
        else:
            existing_results[str(chosen_bandit)] = [results]

        # Write the updated list to the file
        with open(f'model_results_{filename}.json', 'w') as file:
            json.dump(existing_results, file, indent=4)
        if sampling == "thompson":
            sampler.update(chosen_bandit, reward_difference)


    print("Bendt with highest expected improvement:", np.argmax(sampler.wins / (sampler.wins + sampler.losses)))
    print(sampler.wins)
    print(sampler.losses)


if __name__ == "__main__":
    main()
