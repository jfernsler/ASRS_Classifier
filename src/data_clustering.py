import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE

import numpy as np

from jf_nlp.nlp_dataloaders import ASRSRawLoader, ASRSPickleLoader
from jf_nlp.nlp_globals import *
from jf_nlp.nlp_models import get_encoder_model, get_clustering_model

from jf_nlp.utils import pickle_zip, pickle_unzip

import os, random

def create_clusters(num_clusters = 15, save_pickle=True):
    asrs = ASRSRawLoader()

    corpus = asrs.data
    corpus_sentences = corpus['Anomaly'].tolist()

    print("Encode the corpus. This might take a while")
    encoder = get_encoder_model()
    anomaly_embeddings = encoder.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    corpus_embeddings = pd.DataFrame(anomaly_embeddings.cpu())
    print('Encoding complete, shape: {}.'.format(anomaly_embeddings.shape))

    # Generate Clusters
    print("Start clustering")
    clustering_algorithm = get_clustering_model()
    cluster_labels = clustering_algorithm.fit_predict(anomaly_embeddings.cpu())
    corpus['cluster'] = cluster_labels
    print('Clustering complete.')

    # store cluster centers in a dataframe
    cluster_data = pd.DataFrame()
    cluster_centers = clustering_algorithm.cluster_centers_
    cluster_data['cluster'] = np.arange(num_clusters)
    cluster_data['centers'] = cluster_centers.tolist()

    # save to pickle
    if save_pickle:
        pickle_zip(corpus, os.path.join(PICKLE_DATA,'asrs_cluster_labels.pkl.zip'))
        pickle_zip(corpus_embeddings, os.path.join(PICKLE_DATA, 'asrs_label_embeddings.pkl.zip'))
        pickle_zip(cluster_data, os.path.join(PICKLE_DATA, 'asrs_cluster_centers.pkl.zip'))

    print('Corpus shape: ', corpus.shape)
    print('Cluster Data shape: ', cluster_data.shape)
    print('Corpus Embeddings shape: ', corpus_embeddings.shape)


def read_cluster_pickles():
    asrs_data = pd.read_pickle(os.path.join(PICKLE_DATA,'asrs_cluster_labels.pkl'))
    asrs_anomaly_embeddings = pd.read_pickle(os.path.join(PICKLE_DATA, 'asrs_label_embeddings.pkl'))
    asrs_cluster_data = pd.read_pickle(os.path.join(PICKLE_DATA, 'asrs_cluster_centers.pkl'))
    
    print('Corpus shape: ', asrs_data.shape)
    print('Cluster Data shape: ', asrs_cluster_data.shape)
    print('Corpus Embeddings shape: ', asrs_anomaly_embeddings.shape)

    return asrs_data, asrs_anomaly_embeddings, asrs_cluster_data


def print_cluster_samples(corpus, num_clusters=15, sample_size=15):
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
    sample_file = os.path.join(DATA_OUT_DIR, 'cluster_samples.txt')
    with open(sample_file, 'w') as f:
            f.write(sample_string)


def plot_clusters(embeddings, labels, num_clusters = 15, dark_mode = False, perplex = 50, n_iter = 2000, show=False, pt_count=15000):

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


def plot_cluster_counts(asrs_data, show=False):
    counts = asrs_data['cluster'].value_counts().sort_index(ascending=True)

    print('Cluster counts: ')
    print(counts)

    # display a barchart of cluster counts
    plt.figure(figsize=(10, 8))
    plt.bar(counts.index, counts.values)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Cluster Counts')
    plt.savefig(os.path.join(CHART_DIR, 'cluster_counts.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()






def plot_and_sample(asrs_data, asrs_anomaly_embeddings, asrs_cluster_data):
    plot_cluster_counts(asrs_data)
    print_cluster_samples(asrs_data, num_clusters=15, sample_size=15)
    plot_clusters(asrs_anomaly_embeddings, asrs_data['cluster'].values, num_clusters=15, perplex=50)






if __name__ == '__main__':
    #create_clusters()
    asrs_data, asrs_anomaly_embeddings, asrs_cluster_data = read_cluster_pickles()
    plot_and_sample(asrs_data, asrs_anomaly_embeddings, asrs_cluster_data)