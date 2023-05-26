import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import pickle

from jf_nlp import *
from jf_nlp.nlp_dataloaders import ASRSTestLoader, ASRSTrainLoader
from jf_nlp.utils import pickle_unzip, pickle_zip

def get_clean_data(save=False):
    keepers = ['Year', 'Month', 'Flight Phase', 'Reporter Organization', 'Anomaly', 'Narrative']
    data = pd.read_csv(ASRS_DATA, usecols=keepers)
    data.dropna(inplace=True)

    data['Flight Phase'] = data['Flight Phase'].str.lower()
    data['Flight Phase'] = data['Flight Phase'].str.replace('; ', ',')

    data['Anomaly'] = data['Anomaly'].str.lower()
    data['Anomaly'] = data['Anomaly'].str.replace('; ', '. ')
    data['Anomaly'] = data['Anomaly'].str.replace(' / ', ', ')
    data['Anomaly'] = data['Anomaly'].str.rstrip('.') + '.' 
    # adding the flight phase to this column
    data['Anomaly'] = data['Flight Phase'] + ': ' + data['Anomaly']

    data = data[data['Year'] >= 2000]

    if save:
        print('Saving pickle to {}'.format(DATA_DIR))
        data.to_csv(os.path.join(DATA_DIR, 'asrs_data_clean_2000.csv'))

    return data


def save_test_train_data():
    data = pd.read_pickle(os.path.join(PICKLE_DATA,'asrs_cluster_labels.pkl'))
    print(data.head())
    train_data, test_data = train_test_split(data, train_size=0.80, random_state=42)

    pickle_zip(train_data, os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_train.pkl.zip'))
    pickle_zip(test_data, os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_test.pkl.zip'))


def make_anomaly_chart(df, save=False):
    # Group the data by 'Year' and count the number of unique 'Anomaly' values
    total_anomaly_count = df['Anomaly'].nunique()
    unique_anomaly_counts = df.groupby('Year')['Anomaly'].nunique()

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as per your preference
    unique_anomaly_counts.plot(kind='bar', ax=ax)

    # Customize the plot
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Unique Anomaly Values')
    ax.set_title('Unique Anomaly Values by Year (Total Unique Anomalies: {})'.format(total_anomaly_count))
    plt.savefig(os.path.join(CHART_DIR, 'unique_anomalies.png'), bbox_inches='tight', dpi=300)
    plt.show()


def test_pickle_zip():
    train = pickle_unzip(os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_train.pkl.zip'))
    test = pickle_unzip(os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_test.pkl.zip'))
    print(train.shape)
    print(test.shape)


def test_loader():
    test = ASRSTestLoader()
    print(test[43])

    print(test.data.shape)
    print(test.asrs.shape)


def zip_training():
    print('load pre-processed train data...')
    with open(os.path.join(PICKLE_DATA,'asrs_HF_train.pkl'), 'rb') as f:
        train = pickle.load(f)
    
    pickle_zip(train, os.path.join(PICKLE_DATA, 'asrs_HF_train.pkl.zip'))
    
    print('load pre-processed test data...')
    with open(os.path.join(PICKLE_DATA,'asrs_HF_test.pkl'), 'rb') as f:
        test = pickle.load(f)
    
    pickle_zip(test, os.path.join(PICKLE_DATA, 'asrs_HF_test.pkl.zip'))

if __name__ == '__main__':
    # data = get_clean_data(save=True)
    # print(data.shape)
    # print(data.head())
    # make_anomaly_chart(data, save=False)
    #save_test_train_data()
    #test_pickle_zip()
    #test_loader()
    zip_training()