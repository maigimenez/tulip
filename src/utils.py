import re
import numpy as np

def clean_str(text):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def data_augmentation_skip(texts, keep_prob=0.75, mul_value=2):
    more_data = np.zeros((texts.shape[0] * mul_value, texts.shape[1]), dtype=texts.dtype)
    for cur_text, original in enumerate(texts):
        more_data[cur_text] = original
        for i in range(1, mul_value):
            probs = (np.floor(np.random.rand(original.shape[0]) + np.full(original.shape[0], keep_prob))).astype(int)
            new_sample = []
            for pos, i in enumerate(probs):
                if i:
                    new_sample.append(original[pos])
            new_sample.extend([0] * (len(original) - len(new_sample)))
            more_data[cur_text + i] = new_sample
    return more_data


def data_augmentation_dropout(texts, keep_prob=0.75, mul_value=2):
    more_data = np.zeros((texts.shape[0]*mul_value, texts.shape[1]), dtype=texts.dtype)
    for cur_text, original in enumerate(texts):
        more_data[cur_text] = original
        for i in range(1, mul_value):
            probs = (np.floor(np.random.rand(original.shape[0]) + np.full(original.shape[0], keep_prob)))
            new_sample = probs * original
            more_data[cur_text+i] = new_sample
    return more_data


def batch_iter(data, batch_size, num_epochs, num_batches_per_epoch, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)

    print("Num batches per epoch: {} ({})".format(num_batches_per_epoch, data_size))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]