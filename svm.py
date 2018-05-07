import argparse
import sys
import os
import math
import numpy as np
import tensorflow as tf

from learning import INPUT_VECTOR_SIZE, TRAINING_LABEL_POSITIVE, TRAINING_LABEL_NEGATIVE 
from learning import load_training_data

NEG_LABEL = 0
POS_LABEL = 1

class LinkSVM:

    def __init__(self, gpu_num, gpu_mem_frac=.95):
        ModelBase.__init__(self)

        self.weights = self._generate_weights(kernel_size)
        self.labels = self._generate_labels(kernel_size)
        self.bias = self._generate_bias()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, device_count={'GPU': 0, 'CPU': 2}))

        self._create_prediction_graph()

    def _create_architecture(self):
        with tf.device("/gpu:0"):
            self.t_inputs = tf.placeholder(tf.float32, shape=[None, INPUT_VECTOR_SIZE])
            self.t_labels = tf.placeholder(tf.float32, shape=[None, 1])
            
            self.t_weights = tf.Variable(tf.random_normal(shape=[INPUT_VECTOR_SIZE, NUM_CLASSES]))
            self.t_bias = tf.Variable(tf.random_normal(shape=[None, 1])

            self.t_outputs = tf.matmul(self.t_inputs, self.t_weights) - self.t_bias

            self.loss = self.reduce_mean(tf.maximum(0, 1 - tf.multiply(self.t_inputs, self.t_labels)))

    def train(self, training_data, batch_size, num_epochs, learning_rate=.001):
        pos_vecs = training_data[TRAINING_LABEL_POSITIVE] 
        neg_vecs = training_data[TRAINING_LABEL_NEGATIVE]

        feature_vecs = np.array(pos_vecs + neg_vecs, dtype=np.float32)
        labels = np.array([POS_LABEL for _ in pos_vecs] + [NEG_LABEL for _ in neg_vecs], dtype=np.float32)
        
        batches_per_epoch = len(feature_vecs) / batch_size 

        # Shuffle input data
        shuffle_idxs = np.random.permutation(len(feature_vecs))
        feature_vecs = feature_vecs[shuffle_idxs]
        labels = labels[shuffle_idxs]
       
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            for _ in range(batches_per_epoch):
                batch_idxs = np.random.randint(0, len(feature_vecs), batch_size)
                batch_vecs = feature_vecs[batch_idxs]
                batch_labels = labels[batch_idxs]

                feed_dict = {
                    self.inputs : batch_vecs,
                    self.labels : batch_labels
                }

                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

            print(loss)

    def evaluate(self, inputs):
        """
        Parameters
        --------------
        inputs : [np.ndarray]
            A list of float vectors of length 2048,
            represented as numpy arrays
        """
        
        feed_dict = {
            self.t_inputs : inputs
        }

        outputs = self.sess.run(self.t_outputs, feed_dict=feed_dict)
        outputs = outputs.flatten()

        print(outputs)
    
    def get_weights(self):
        weights = sess.run(self.t_weights, feed_dict={})
        return weights

    def get_bias(self):
        bias = sess.run(self.t_bias, feed_dict={})
        return bias

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LinkSVM')
    parser.add_argument('-t', '--training_path', type=str, help="The path to a directory containing model training data")

    args = parser.parse_args()

    pos_max = 1000
    neg_max = 1500

    training_data = load_training_data(args.training_path, pos_max, neg_max)
    net = LinkNet(gpu_num=0)
    net.train(training_data, 30, 100)
    print(net.evaluate(training_data[TRAINING_LABEL_POSITIVE][0]))
    print(net.evaluate(training_data[TRAINING_LABEL_NEGATIVE][0]))
