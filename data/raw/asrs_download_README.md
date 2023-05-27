## ASRS Raw Dataset

This dataset is sourced from:

https://www.kaggle.com/datasets/andrewda/aviation-safety-reporting-system

it's a 500+Mb .csv file. You will need this if you wish to re-run the all of the pre-processing and clustering. Download it, unzip it, and place it in PROJECT/data/raw with the name asrs.csv.

Then you can re-run asrs_create_clusters.py and the asrs_preprocess.py to create the clustering labels and rebuild the embeddings into an HF dataset. 