import os
import sys
import csv
import time
import json
import datetime
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

import data_helper
from rnn_classifier import rnn_clf

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    error = "Please install scikit-learn."
    print(str(e) + ': ' + error)
    sys.exit()

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Model choices
tf.flags.DEFINE_string('clf', 'blstm', "Type of classifiers. Default: blstm. You have two choices: [lstm, blstm]")

# Data parameters
tf.flags.DEFINE_string('data_file', None, 'Data file path')
tf.flags.DEFINE_string('stop_word_file', None, 'Stop word file path')
tf.flags.DEFINE_string('language', 'en', "Language of the data file. You have two choices: [ch, en]")
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 0, 'Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('test_size', 0.2, 'Cross validation test size')

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 100, 'Word embedding size.')
tf.flags.DEFINE_integer('hidden_size', 128, 'Number of hidden units in the LSTM cell')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of the LSTM cells.')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')  # All
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')  # All
tf.flags.DEFINE_float('l2_reg_lambda', 0.0001, 'L2 regularization lambda')  # All

# Training parameters
tf.flags.DEFINE_integer('batch_size', 20, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 0.95, 'Learning rate decay rate. Range: (0, 1]')  # Learning rate decay
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps')  # Learning rate decay
tf.flags.DEFINE_integer('evaluate_every_steps', 100, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 1000, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')

FLAGS = tf.app.flags.FLAGS

def preprocess_train():
    x_text, y, lengths = data_helper.load_data_and_labels('train/pos.txt', 'train/neg.txt')
    max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = 300
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    return x, y, vocab_processor, lengths

def preprocess_dev(vocab_processor_train):
    x_test, y, lengths = data_helper.load_data_and_labels('dev/pos.txt', 'dev/neg.txt')

    word_to_idx = vocab_processor_train.vocabulary_._mapping
    x = []
    for sentence in x_test:
        sent_to_idx = []
        for word in sentence.split(" "):
            if word in word_to_idx:
                idx = word_to_idx[word]
            else:
                idx = 0
            sent_to_idx.append(idx)
        x.append(sent_to_idx)

    max_length = max(map(len, [sent for sent in x]))

    for i in range(len(x)):
        x[i] = np.pad(x[i], (0, max_length - len(x[i])), 'constant', constant_values=(0,0))[0:300]

    return x, y, lengths

def preprocess_test(vocab_processor_train):
    x_test, y, lengths = data_helper.load_data_and_labels('test/pos.txt', 'test/neg.txt')

    word_to_idx = vocab_processor_train.vocabulary_._mapping
    x = []
    for sentence in x_test:
        sent_to_idx = []
        for word in sentence.split(" "):
            if word in word_to_idx:
                idx = word_to_idx[word]
            else:
                idx = 0
            sent_to_idx.append(idx)
        x.append(sent_to_idx)

    max_length = max(map(len, [sent for sent in x]))

    for i in range(len(x)):
        x[i] = np.pad(x[i], (0, max_length - len(x[i])), 'constant', constant_values=(0,0))[0:300]

    return x, y, lengths

def train(x_train, y_train, vocab_processor, x_dev, y_dev, lengths_train, lengths_dev, lengths_test, x_test, y_test):

    with tf.Session() as sess:
        model = rnn_clf(FLAGS)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                   decay_rate=FLAGS.decay_rate,
                                                   decay_steps=FLAGS.decay_steps,
                                                   staircase=True,
                                                   global_step=global_step)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoints_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoint)
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        FLAGS.vocab_size = len(vocab_processor.vocabulary_._mapping)
        FLAGS.max_length = vocab_processor.max_document_length

        sess.run(tf.global_variables_initializer())

        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            saver.restore(sess, checkpoint_state.model_checkpoint_path)
            print('Restore model sucess')
        else:
            print('No model to load at {}'.format(checkpoint_dir))

        def train_step(x_batch, y_batch, lengths_batch):
            feed_dict = {model.input_x: x_batch,
                         model.input_y: y_batch,
                         model.keep_prob: FLAGS.keep_prob,
                         model.batch_size: FLAGS.batch_size,
                         model.sequence_length: lengths_batch}

            _, step, loss, accuracy = sess.run(
                [train_op, global_step, model.cost, model.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def dev_step(x_batch, y_batch, lengths_batch):
            feed_dict = {model.input_x: x_batch,
                         model.input_y: y_batch,
                         model.keep_prob: 1,
                         model.batch_size: len(y_batch),
                         model.sequence_length: lengths_batch}

            step, loss, accuracy = sess.run(
                [global_step, model.cost, model.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        batches = data_helper.batch_iter(
            x_train, y_train, lengths_train, FLAGS.batch_size, FLAGS.num_epochs)

        for epoch in range(0, 20):

	        for batch in batches:
	            x_batch, y_batch, lengths_batch = batch

	            train_step(x_batch, y_batch, lengths_batch)
	            current_step = tf.train.global_step(sess, global_step)
	            if current_step % FLAGS.evaluate_every_steps == 0:
	                print("\nEvaluation:")
	                dev_step(x_dev, y_dev, lengths_dev)
	                dev_step(x_test, y_test, lengths_test)
	                print("")
	            if current_step % FLAGS.save_every_steps == 0:
	                path = saver.save(sess, checkpoints_prefix, global_step=current_step)
	                print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor_train, lengths_train = preprocess_train()
    x_dev, y_dev, lengths_dev = preprocess_dev(vocab_processor_train)
    x_test, y_test, lengths_test = preprocess_test(vocab_processor_train)
    train(x_train, y_train, vocab_processor_train, x_dev, y_dev, lengths_train, lengths_dev, lengths_test, x_test, y_test)

if __name__ == "__main__":
    tf.app.run()