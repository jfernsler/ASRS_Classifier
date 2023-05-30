####################################################################################################
# This file contains functions to manage the checkpoint files for the ASRS DistilBERT model.
# plot_loss() - plot the loss from the training log
# push_model() - push the model and tokenizer to the HuggingFace Hub
# get_model() - get the model and tokenizer from the HuggingFace Hub as a test
####################################################################################################

import json
from jf_nlp.nlp_globals import *

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def plot_loss():
    # load the json file
    log_file = os.path.join(MODEL_PATH_TUNED, 'checkpoint-'+str(MODEL_BEST_CHECKPOINT), 'trainer_state.json')
    with open(log_file) as f:
        data = json.load(f)
    # print the data
    log_history = data['log_history']
    loss = [[x['loss'], x['epoch']] for x in log_history if 'loss' in x]

    print(len(loss))
    print(loss)
    
    # create graph of loss with loss[0] on the y axis and loss[1] on the x axis
    import matplotlib.pyplot as plt
    plt.plot([x[1] for x in loss], [x[0] for x in loss])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('ASRS DistilBERT Tuned Loss')
    plt.savefig(os.path.join(CHART_DIR, 'asrs_distilbert_tuned_loss.png'), dpi=300, bbox_inches='tight')
    plt.show()
    

def push_model():
    """Push the model and tokenizer to the HuggingFace Hub. Need to input access token."""
    access_token = "hf_..."

    model_path = os.path.join(MODEL_PATH_TUNED, 'checkpoint-' + str(MODEL_BEST_CHECKPOINT))

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.push_to_hub('ASRS_distilbert-base-uncased', use_auth_token=access_token)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.push_to_hub('ASRS_distilbert-base-uncased', use_auth_token=access_token)
    print(model)

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("jfernsler/ASRS_distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("jfernsler/ASRS_distilbert-base-uncased")    

    print(model)


if __name__ == '__main__':
    #plot_loss()
    #push_model()
    #get_model()
    pass