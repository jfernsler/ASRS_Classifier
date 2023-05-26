from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .nlp_globals import *

def plot_cluster_counts(asrs_data, name='cluster_counts', show=False):
    """Plots a bar chart of cluster counts."""
    counts = asrs_data['cluster'].value_counts().sort_index(ascending=True)
    title = name.split('_')
    title = ' '.join(title)
    
    # display a barchart of cluster counts
    plt.figure(figsize=(10, 8))
    plt.bar(counts.index, counts.values)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(os.path.join(CHART_DIR, f'{name}.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()

def make_matrix(y_true, y_pred, classes, title, figure_path):
    """Makes a confusion matrix from the given data."""
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                            index = [classes],
                            columns = [classes])
    plt.figure(figsize = (16,12))
    plt.subplots_adjust(bottom=0.25)
    plt.title(title)
    hm = sn.heatmap(df_cm, annot=True, linewidths=.5, cmap='plasma', fmt='.2f', linecolor='grey')
    hm.set(xlabel='Predicted', ylabel='Truth')
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")

def print_results(index, text, y, yhat, confidence):
    """Prints the results of a single inference."""
    print('\n', '='*80)
    print('\nNarrative at index {}:\n'.format(index), text)
    print('\nTrue Label:', y)
    print('Predicted Label: {} (Confidence: {:.2f})'.format(yhat, confidence))
    print('\n', '='*80)