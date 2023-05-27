from transformers import AutoTokenizer
from datasets import Dataset, Features, Value, ClassLabel

from jf_nlp.nlp_dataloaders import ASRSTestLoader, ASRSTrainLoader
from jf_nlp.nlp_globals import *
from jf_nlp.utils import pickle_zip

model_checkpoint = "distilbert-base-uncased"
batch_size = 16
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def rebuild_encodings():
    """Load the train and test data - then rebuild and store the encodings for huggingface datasets"""
    test = ASRSTestLoader()
    train = ASRSTrainLoader()

    # identify the features
    asrs_features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=15),
        })
    
    test_dataset = Dataset.from_pandas(test.data[['text','label']], features=asrs_features)
    train_dataset = Dataset.from_pandas(train.data[['text','label']], features=asrs_features)

    tokenized_test = test_dataset.map(preprocess_function)
    pickle_zip(tokenized_test, os.path.join(PICKLE_DATA,'asrs_HF_test.pkl.zip'))

    tokenized_train = train_dataset.map(preprocess_function)
    pickle_zip(tokenized_train, os.path.join(PICKLE_DATA,'asrs_HF_train.pkl.zip'))

if __name__ == '__main__':
    rebuild_encodings()