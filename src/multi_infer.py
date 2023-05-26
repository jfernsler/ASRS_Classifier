from transformers import pipeline

from jf_nlp.nlp_dataloaders import ASRSTestLoader
from jf_nlp.nlp_output_utils import make_matrix, print_results
from jf_nlp.nlp_globals import *

import os, random

def infer_many(count = 10, display = True, chart = False, checkpoint=MODEL_BEST_CHECKPOINT):
    """Infer many randomly selected narratives."""
    asrs = ASRSTestLoader()
    idx_list = random.sample(range(len(asrs)), count)

    model_path = os.path.join(MODEL_PATH_TUNED, 'checkpoint-{}'.format(checkpoint))
    classifier = pipeline("sentiment-analysis", model=model_path)

    y_list = list()
    yhat_list = list()
    correct_count = 0
    random_count = 0

    for n, idx in enumerate(idx_list):
        r_marker = '_'
        element = asrs[idx]

        y_list.append(int(element['label']))

        result = classifier(element['text'], padding=True, truncation=True, max_length=512)

        yhat_list.append(int(result[0]['label']))
        confidence = result[0]['score']

        if y_list[-1] == yhat_list[-1]:
            correct_count += 1
            r_marker = '+'
        
        if y_list[-1] == random.randrange(14):
            random_count += 1

        if display:
            print_results(idx, element['text'], y_list[-1], yhat_list[-1], confidence)
        else:
            print(r_marker, end='', flush=True)
            if (n+1) % 100 == 0:
                print('||{}||'.format(n+1), end='', flush=True)

    print()
    print(f'Correct: {correct_count} / {len(idx_list)}, Percentage: {correct_count / float(len(idx_list)) * 100:.2f}')
    print(f'Random: {random_count} / {len(idx_list)}, Percentage: {random_count / float(len(idx_list)) * 100:.2f}')
    print()

    if chart:
        chart_path = os.path.join(CHART_DIR, 'confusion_matrix_{}_samples.png'.format(len(idx_list)))
        classes = ['00', '01','02','03','04','05','06','07','08','09','10','11','12','13','14']
        title = 'Confusion Matrix, {} Samples'.format(len(idx_list))
        #      y_true, y_pred, classes, title, figure_path
        make_matrix(y_true=y_list, y_pred=yhat_list, classes=classes, title=title, figure_path=chart_path)


def infer_one(idx=None, checkpoint=MODEL_BEST_CHECKPOINT):
    """Infer a single narrative."""
    asrs = ASRSTestLoader()

    model_path = os.path.join(MODEL_PATH_TUNED, 'checkpoint-{}'.format(checkpoint))
    classifier = pipeline("sentiment-analysis", model=model_path)

    if idx is None:
        idx = random.randrange(len(asrs))

    element = asrs[idx]
    result = classifier(element['text'], padding=True, truncation=True, max_length=512)
    print_results(idx, element['text'], int(element['label']), int(result[0]['label']), result[0]['score'])

    

if __name__ == '__main__':
    infer_many(20)
    #infer_one()