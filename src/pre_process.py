# need to:
# 1. split the data into train and test sets
# 2. train the model
# 3. test the model
# 4. save the model
# 5. load the model
#
#import transformers
from transformers import AutoTokenizer
from datasets import Dataset, Features, Value, ClassLabel

from jf_nlp.nlp_dataloaders import ASRSPickleLoader, ASRSTestLoader, ASRSTrainLoader
from jf_nlp.nlp_output_utils import plot_cluster_counts
from jf_nlp.nlp_globals import *
from jf_nlp.utils import pickle_zip, pickle_unzip
from sklearn.model_selection import train_test_split

import pickle

import pandas as pd

model_checkpoint = "distilbert-base-uncased"
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def build_encodings(train_ratio = 0.80, graphics = False):
    asrs = ASRSPickleLoader()

    # split the data into train and test sets
    train, test = train_test_split(asrs.data, train_size=train_ratio, random_state=42)

    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    if graphics:
        plot_cluster_counts(train, name='Train_Clusters')
        plot_cluster_counts(test, name='Test_Clusters')

    encoded_train = train['Narrative'].map(preprocess_function)
    encoded_test = test['Narrative'].map(preprocess_function)

    encoded_train.to_pickle(os.path.join(PICKLE_DATA,'asrs_encoded_train.pkl'))
    encoded_test.to_pickle(os.path.join(PICKLE_DATA,'asrs_encoded_test.pkl'))

    print(encoded_train.iloc[0])
    pass

def rebuild_encodings():
    test = ASRSTestLoader()
    train = ASRSTrainLoader()

    asrs_features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=15),
        })
    
    test_dataset = Dataset.from_pandas(test.data[['text','label']], features=asrs_features)
    train_dataset = Dataset.from_pandas(train.data[['text','label']], features=asrs_features)

    tokenized_test = test_dataset.map(preprocess_function)
    tokenized_train = train_dataset.map(preprocess_function)

    with open(os.path.join(PICKLE_DATA,'asrs_HF_test.pkl'), 'wb') as f:
        pickle.dump(tokenized_test, f)
    with open(os.path.join(PICKLE_DATA,'asrs_HF_train.pkl'), 'wb') as f:
        pickle.dump(tokenized_train, f)

    #small_test_dataset = tokenized_test.shuffle(seed=42).select(range(1000))
    #small_train_dataset = tokenized_train.shuffle(seed=42).select(range(1000))


def test_encodings(train_ratio = 0.08):
    asrs = ASRSPickleLoader()
    # split the data into train and test sets
    train, test = train_test_split(asrs.data, train_size=train_ratio, random_state=42)

    test_position = 123

    print(test.iloc[test_position]['Narrative'])
    print(test.iloc[test_position]['cluster'])
    print()
    print(tokenizer(test.iloc[test_position]['Narrative'], padding="max_length", truncation=True))
    print()
    test_embeddings = pd.read_pickle(os.path.join(PICKLE_DATA, 'asrs_encoded_test.pkl'))
    print(test_embeddings.iloc[test_position])

def test_loaders():
    #train = ASRSTrainLoader()
    test = ASRSTestLoader()

    print(test[0])
    pass

if __name__ == '__main__':
    rebuild_encodings()