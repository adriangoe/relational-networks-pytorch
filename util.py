# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from sklearn import preprocessing
import h5py
import numpy as np


lb = preprocessing.LabelBinarizer()
lb.fit(['right', 'blue', 'circle', 'left', 'bottom', 'yellow',
        'square', 'green', 'red', 'top', 'gray'])


class NotSoCLEVRDataset(Dataset):
    '''Loader class for the Not So CLEVR Dataset'''

    def __init__(self, csv_file, img_file, transform=None):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_file (string): Path to the h5 file with image vectors.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self.questions = pd.read_csv(csv_file)

        # Load all files into memory since we've seen
        # errors when loading from h5 on GPU.
        f = h5py.File(img_file, 'r')
        group_key = list(f.keys())[0]
        self.images = list(f[group_key])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            img = self.transform(self.images[idx])
        else:
            img = self.images[idx]

        sample = {'image': img,
                  'task': self.questions.iloc[idx, 1],
                  'question': literal_eval(self.questions.iloc[idx, 2]),
                  'answer': lb.transform([self.questions.iloc[idx, 3]]
                                         ).argmax(),
                  'type': self.questions.iloc[idx, 4]}

        return sample
