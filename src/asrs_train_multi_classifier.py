################################################################################
# This is where the fine-tuning happens.
# The model is trained on the ASRS dataset
# The model is then saved to the MODEL folder
# train() is the main function

from jf_nlp.nlp_globals import *
from jf_nlp.utils import pickle_unzip

import evaluate
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

import numpy as np

import os, gc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluate.load('accuracy').compute(predictions=predictions, references=labels)


def train(batch_size=8, reduce=False, epochs=8, model_checkpoint='distilbert-base-uncased'):
    gc.collect()
    torch.cuda.empty_cache()

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # load the data
    print('load pre-processed training data...')
    train = pickle_unzip(ASRS_HF_TRAIN)
    print('load pre-processed test data...')
    test = pickle_unzip(ASRS_HF_TEST)

    if reduce:
        print('reducing the dataset by 10x...')
        train = train.shuffle(seed=42).select(range(len(train)//10))
        test = test.shuffle(seed=42).select(range(len(test)//10))

    print('define the model...')
    id2label = dict()
    label2id = dict()
    for l in range(15):
        id2label[l] = str(l)
        label2id[str(l)] = l

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)

    print('define the trainer...')
    training_args = TrainingArguments(
        output_dir=MODEL_PATH_TUNED,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        optim="adamw_torch"
        #metric_for_best_model="accuracy",
        )
    
    print('define the trainer...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )

    print('training...')
    trainer.train()

def check_cuda():
    print('list torch devices:', torch.cuda.device_count())
    print('cuda available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    print('cuda device:', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0).total_memory)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    #check_cuda()
    train()