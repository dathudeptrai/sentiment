import tensorflow as tf
import time
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
import numpy as np
from tensorflow.contrib import rnn

class rnn_clf(object):
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # Word embedding
        with tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding',
                                        shape=[self.vocab_size, self.hidden_size],
                                        initializer=xavier_initializer(),
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        self.inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        self.outputs = self.bi_lstm() # [batch_size, max_time, 256]

        self.attention_score = tf.nn.softmax(tf.layers.dense(self.outputs, units=1))
        self.attention_out = tf.matmul(tf.transpose(self.outputs, perm=[0, 2, 1]), self.attention_score)
        self.attention_out = tf.squeeze(self.attention_out, -1) # [batch_size, 256]

        with tf.name_scope('softmax'):

            softmax_w = tf.get_variable('softmax_w', shape=[2 * self.hidden_size, self.num_classes], dtype=tf.float32, 
                                        initializer=xavier_initializer(),
                                        regularizer=l2_regularizer(scale=0.2))
            softmax_b = tf.get_variable('softmax_b', shape=[self.num_classes], dtype=tf.float32, 
                                        initializer=xavier_initializer(),
                                        regularizer=l2_regularizer(scale=0.2))

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)
            
            self.logits = tf.matmul(self.attention_out, softmax_w) + softmax_b

            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

        # Loss
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()

            for tv in tvars:
                if 'kernel' in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    def bi_lstm(self):
    	def lstm_fw():
    		cell_fw = rnn.LSTMBlockCell(self.hidden_size)
    		cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
    		return cell_fw

    	cell_fw = rnn.MultiRNNCell([lstm_fw() for _ in range(self.num_layers)], state_is_tuple=True)
    	cell_bw = rnn.MultiRNNCell([lstm_fw() for _ in range(self.num_layers)], state_is_tuple=True)
    	self._initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
    	self._initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)

    	with tf.variable_scope('Bi-LSTM'):
    		outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
    			                                       cell_bw,
    			                                       inputs=self.inputs,
    			                                       initial_state_fw=self._initial_state_fw,
    			                                       initial_state_bw=self._initial_state_bw,
    			                                       sequence_length=self.sequence_length)

    	output = tf.concat(outputs, -1)

    	return output