import numpy as np
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " is", string)
	string = re.sub(r"\'ve", " have", string)
	string = re.sub(r"n\'t", " not", string)
	string = re.sub(r"\'re", " are", string)
	string = re.sub(r"\'d", " would", string)
	string = re.sub(r"\'ll", " will", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)

	#string = " ".join([word for word in string.split() if word not in set(stopwords.words('english'))])
	return string.strip().lower()

def load_data_and_labels(positive_data_file, negative_data_file):
	positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
	negative_examples = [s.strip() for s in negative_examples]

	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]

	positive_labels = [1 for _ in positive_examples]
	negative_labels = [0 for _ in negative_examples]
	y = positive_labels + negative_labels
	shuffle_indices = np.random.permutation(np.arange(len(x_text)))
	x_text = np.array(x_text)[shuffle_indices]
	y = np.array(y)[shuffle_indices]
	lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in x_text])))
	return [x_text, y, lengths]

def batch_iter(data, labels, lengths, batch_size, num_epochs):

    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length