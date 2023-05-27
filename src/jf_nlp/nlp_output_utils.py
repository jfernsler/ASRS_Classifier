from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

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


def plot_clusters(embeddings, labels, num_clusters = 15, dark_mode = False, perplex = 50, n_iter = 2000, show=False, pt_count=15000):
    """Plots a 3D scatter plot of the clusters. Requires the embeddings and labels from the model."""
    embeddings = embeddings[:pt_count]
    labels = labels[:pt_count]

    print('Plotting {} points'.format(embeddings.shape[0]))

    # Reduce dimensionality of embeddings
    reducer = TSNE(n_components=3, perplexity=perplex)  # Use t-SNE for dimensionality reduction
    reduced_embeddings = reducer.fit_transform(embeddings)    

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define custom markers and colors for each cluster
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'h', 'x', '*', 'H', '+', 'd', '|', '_', '8']
    # Define colors that will work on a white backgound:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf', '#b5bdff', '#ffbdbd', '#c7e2c7', '#ffb5b5', '#e5e5e5', '#ffdbb5']

    if dark_mode:
        colors = ['#FF5252', '#FF4081', '#E040FB', '#7C4DFF', '#536DFE', '#448AFF', '#40C4FF', '#18FFFF',
                '#64FFDA', '#69F0AE', '#B2FF59', '#EEFF41', '#FFFF00', '#FFD740', '#FFAB40', '#FF6E40',
                '#FF4081', '#8D6E63']

        # Set a dark gray background color
        fig.patch.set_facecolor('#666666')
        ax.set_facecolor('#666666')

    # Iterate over each cluster and plot the points with corresponding markers and colors
    for i in range(num_clusters):
        points_in_cluster = reduced_embeddings[labels == i]
        ax.scatter(
            points_in_cluster[:, 0],
            points_in_cluster[:, 1],
            points_in_cluster[:, 2],
            marker=markers[i % len(markers)],  # Cycle through markers if more clusters than markers
            color=colors[i % len(colors)],  # Cycle through colors if more clusters than colors
            s=30,  # Adjust the size of the points as per your preference
            label=f'Cluster {i+1}'
        )

    ax.set_title('Cluster Visualization : Perplexity {}'.format(perplex))
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.legend()
    plt.savefig(os.path.join(CHART_DIR, 'cluster_viz_{}.png'.format(perplex)), dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def print_cluster_samples(corpus, num_clusters=15, sample_size=15, save=True):
    """Prints a sample of anomalies from each cluster."""
    sample_string = ''
    sample_string += 'Printing {0} samples from each of {1} clusters'.format(sample_size, num_clusters) + '\n\n'
    for i in range(num_clusters):
        sample_string += 'Cluster {}:\n'.format(i)
        cluster = corpus[corpus['cluster'] == i]
        # get random sample from cluster
        samples = cluster.sample(sample_size)
        for sample in samples['Anomaly']:
            sample_string += sample + '\n'
        sample_string += '\n'
    
    print()
    print('#'*80)
    print(sample_string)
    print('#'*80)
    print()

    # save this to a file
    if save:
        sample_file = os.path.join(DATA_OUT_DIR, 'cluster_samples.txt')
        with open(sample_file, 'w') as f:
                f.write(sample_string)


def make_anomaly_chart(asrs_data, save=False):
    """Creates a bar chart of the number of unique anomaly values by year.
    Requires a dataframe with a 'Year' column and an 'Anomaly' column.
    """
    # Group the data by 'Year' and count the number of unique 'Anomaly' values
    total_anomaly_count = asrs_data['Anomaly'].nunique()
    unique_anomaly_counts = asrs_data.groupby('Year')['Anomaly'].nunique()

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as per your preference
    unique_anomaly_counts.plot(kind='bar', ax=ax)

    # Customize the plot
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Unique Anomaly Values')
    ax.set_title('Unique Anomaly Values by Year (Total Unique Anomalies: {})'.format(total_anomaly_count))
    if save:
        plt.savefig(os.path.join(CHART_DIR, 'unique_anomalies.png'), bbox_inches='tight', dpi=300)
    plt.show()