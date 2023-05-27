import pandas as pd
import torch
import numpy as np

from .nlp_globals import *
from .utils import pickle_unzip, pickle_zip

import os

class ASRSRawLoader():
    """Loads the original ASRS data from the csv file and cleans it up a bit."""
    def __init__(self, start_year=2000, load_pickle=True, save_pickle=True):

        # load from pickle if it exists
        if os.path.exists(ASRS_DATA_CLEAN) and load_pickle:
            self.data = self._load_pickle(ASRS_DATA_CLEAN)
            return

        keepers = ['ACN', 'Year', 'Month', 'Flight Phase', 'Reporter Organization', 'Anomaly', 'Narrative']
        self.data = pd.read_csv(ASRS_RAW_DATA, usecols=keepers)
        self.data.dropna(inplace=True)

        self.data['Narrative'] = self.data['Narrative'].str.lower()
        self.data['Narrative'] = self.data['Narrative'].str.replace('; ', ', ')
        self.data['Narrative'] = self.data['Narrative'].str.rstrip('.') + '.' 

        self.data['Flight Phase'] = self.data['Flight Phase'].str.lower()
        self.data['Flight Phase'] = self.data['Flight Phase'].str.replace('; ', ', ')

        self.data['Anomaly'] = self.data['Anomaly'].str.lower()
        self.data['Anomaly'] = self.data['Anomaly'].str.replace('; ', '. ')
        self.data['Anomaly'] = self.data['Anomaly'].str.replace(' / ', ', ')
        self.data['Anomaly'] = self.data['Anomaly'].str.rstrip('.') + '.' 

        self.data = self.data[self.data['Year'] >= start_year]
        
        # save to pickle for later
        if save_pickle:
            self.save_pickle()

    def save_pickle(self, filename=ASRS_DATA_CLEAN):
        print('Saving pickle to {}'.format(filename))
        pickle_zip(self.data, filename)

    def _load_pickle(self, filename=ASRS_DATA_CLEAN):
        print('Loading pickle from {}'.format(filename))
        self.data = pickle_unzip(filename)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        year = self.data.iloc[idx]['Year']
        month = self.data.iloc[idx]['Month']
        anomaly = self.data.iloc[idx]['Anomaly']
        narrative = self.data.iloc[idx]['Narrative']
        return {'year': year, 'month': month, 'anomaly': anomaly, 'narrative': narrative}

    def get_random_sample(self, count=10):
        return self.data.sample(count).reset_index(drop=True)


class ASRSLoader():
    """Loads the ASRS data from the pickle file and creates a column reduced data set."""
    def __init__(self, pickle_file='asrs_cluster_labels.pkl.zip'):

        self.pickle_file = os.path.join(PICKLE_DATA, pickle_file)
        self.asrs = pickle_unzip(self.pickle_file)
        #self.asrs = pd.read_pickle(os.path.join(PICKLE_DATA,'asrs_cluster_labels.pkl'))
        self.data = self.asrs.copy()
        self.reduce_columns()


    def reduce_columns(self):
        self.data.drop(columns = ['ACN', 'Year', 'Month', 'Flight Phase', 'Reporter Organization', 'Anomaly'], inplace=True)
        self.data.rename(columns={'cluster': 'label', 'Narrative': 'text'}, inplace=True)
        self.data['label'] = self.data['label'].astype(np.int64)
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        label = self.data.iloc[idx]['label']
        label = torch.from_numpy(np.array(label).astype(np.float32))
        text = self.data.iloc[idx]['text']
        raw_dict = {'label': label, 'text': text, 'idx': idx}
        return raw_dict


class ASRSTrainLoader(ASRSLoader):
    """Subclass of ASRSLoader that loads the training data."""
    def __init__(self):
        print('Loading ASRS Train Data')
        super().__init__(pickle_file='asrs_data_clean_2000_train.pkl.zip')


class ASRSTestLoader(ASRSLoader):
    """Subclass of ASRSLoader that loads the test data."""
    def __init__(self):
        #super().__init__()
        print('Loading ASRS Test Data')
        super().__init__(pickle_file='asrs_data_clean_2000_test.pkl.zip')


if __name__ == '__main__':
    asrs = ASRSTestLoader()
    # print 10 random samples
    print(asrs.data.iloc[100])
    print(asrs.data.shape)

