## This directory could contain quite a bit of data

* ```asrs_cluster_labels.pkl.zip```
* ```asrs_data_clean_2000_test.pkl.zip```
* ```asrs_data_clean_2000_train.pkl.zip```
    * result of creating an ```ASRSRawLoader()```
    * it will load the raw .csv file (if you have it) then clean it up, split it, and pickle it.
* ```asrs_cluster_centers.pkl.zip```
* ```asrs_cluster_labels.pkl.zip```
* ```asrs_label_embeddings.pkl.zip```
    * result of ```asrs_create_clusters.py```
    * needed to plot the clusters
* ```asrs_HF_test.pkl.zip```
* ```asrs_HF_train.pkl.zip```
    * result of ```asrs_preprocess.py```
    * contains all the data PLUS narrative embeddings, will be quite large.
    * needed to train the model
