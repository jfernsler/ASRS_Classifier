from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def get_encoder_model():
    return SentenceTransformer('all-distilroberta-v1')

def get_clustering_model():
    return KMeans(n_clusters=15, n_init='auto')

def get_classifer_model():
    return None