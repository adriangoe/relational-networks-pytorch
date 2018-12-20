# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from sklearn import preprocessing
import h5py
import torch
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

        f = h5py.File(img_file, 'r')
        group_key = list(f.keys())[0]
        self.images = f[group_key]
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
                  'target': lb.transform([self.questions.iloc[idx, 3]]
                                         ).argmax(),
                  'answer': self.questions.iloc[idx, 3],
                  'type': self.questions.iloc[idx, 4]}

        return sample


class TextUtil():
    '''To process text-sentences for use in an LSTM
    '''

    def __init__(self, text_csv, text_column):
        self.vocab = set(' '.join(pd.read_csv(text_csv)[text_column].tolist()
                                  ).split(' '))
        self.max_len = max([len(s.split(' ')) for s in
                            pd.read_csv(text_csv)[text_column].tolist()])
        self.vocab_size = len(self.vocab)
        self.word_to_ix = {word: i for i, word
                           in enumerate(self.vocab)}

    def string_to_vec(self, string):
        '''Turn a sentence into a vector
        of vocabulary indeces.
        '''
        # set all to stop-word
        out = np.ones((self.max_len)) * self.vocab_size
        for i, w in enumerate(string.split(' ')):
            out[i] = self.word_to_ix[w]
        return torch.tensor(out, dtype=torch.long)
