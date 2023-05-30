####################################################################################################
# This script is used to create the clusters for the ASRS dataset
#    First embeddings are created for each anomaly using the encoder model
#    Then the clustering model is used to create the clusters
#    Then the clusters are visualized using the TSNE algorithm

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from jf_nlp.nlp_dataloaders import ASRSRawLoader
from jf_nlp.nlp_globals import *
from jf_nlp.nlp_models import get_encoder_model, get_clustering_model

from jf_nlp.utils import pickle_zip

import os

def create_clusters(num_clusters = 15, save_pickle=True):
    """Create clusters for the ASRS dataset using the encoder and clustering models"""
    asrs = ASRSRawLoader()

    corpus = asrs.data
    corpus_sentences = corpus['Anomaly'].tolist()

    print("Embedding the anomalies. This might take a while")
    encoder = get_encoder_model()
    anomaly_embeddings = encoder.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    corpus_embeddings = pd.DataFrame(anomaly_embeddings.cpu())
    print('Embedding complete, shape: {}.'.format(anomaly_embeddings.shape))

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
        print('Saving pickles to {}'.format(PICKLE_DATA))
        pickle_zip(corpus, os.path.join(PICKLE_DATA,'asrs_cluster_labels.pkl.zip'))

        train_data, test_data = train_test_split(corpus, train_size=0.80, random_state=42)
        pickle_zip(train_data, os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_train.pkl.zip'))
        pickle_zip(test_data, os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_test.pkl.zip'))

        #pickle_zip(corpus_embeddings, os.path.join(PICKLE_DATA, 'asrs_label_embeddings.pkl.zip'))
        #pickle_zip(cluster_data, os.path.join(PICKLE_DATA, 'asrs_cluster_centers.pkl.zip'))

    # feedback
    print()
    print('Corpus shape: ', corpus.shape)
    print('Cluster Data shape: ', cluster_data.shape)
    print('Corpus Embeddings shape: ', corpus_embeddings.shape)
    print()

if __name__ == '__main__':
    create_clusters()