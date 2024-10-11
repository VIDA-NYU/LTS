import pandas as pd
from fine_tune import BertFineTuner
import unittest


class TestProcessData(unittest.TestCase):

    df = pd.read_csv("llama_data_labeled.csv")
    validation= pd.read_csv("validation.csv")
    validation["text"] = validation["title"]
    trainer = BertFineTuner("bert-base-uncased", df, validation)
    results = trainer.train_data(df)
    print(results)



if __name__ == '__main__':
    unittest.main()
