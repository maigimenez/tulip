import numpy as np
from utils import clean_str
import pandas as pd


class Data:
    def __init__(self, text, polarity):
        self._text = clean_str(text)
        self._polarity = polarity

    def __repr__(self):
        return '({}): {}'.format(self._polarity, self._text)

    @property
    def text(self):
        return self._text

    @property
    def polarity(self):
        return self._polarity


class Corpus:
    def __init__(self, *args):
        self._paths = args
        self._data = []

    @property
    def data(self):
        return self._data

    def load_mr(self):
        neg_datapath = self._paths[0]
        pos_datapath = self._paths[1]
        with open(pos_datapath) as pos_file:
            for review in pos_file:
                self._data.append(Data(review.strip(), [0, 1]))
        with open(neg_datapath) as neg_file:
            for review in neg_file:
                self._data.append(Data(review.strip(), [1, 0]))

    def load_sst(self):
        data_set_csv_path = self._paths[0]
        sentences = pd.DataFrame.from_csv(data_set_csv_path)['TEXT'].tolist()
        polarities = pd.DataFrame.from_csv(data_set_csv_path)['POLARITY'].tolist()
        assert len(sentences) == len(polarities)
        for sentence, polarity in zip(sentences, polarities):
            if polarity > 0.5:
                self._data.append(Data(sentence.strip(), [0, 1]))
            else:
                self._data.append(Data(sentence.strip(), [1, 0]))
        assert len(self._data) == len(sentences)

    def get_texts(self):
        return [data.text for data in self._data]

    def get_labels(self):
        return np.array([data.polarity for data in self._data])
