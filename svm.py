import argparse
import sys
import os
import math
import numpy as np
import tensorflow as tf

from learning import INPUT_VECTOR_SIZE, TRAINING_LABEL_POSITIVE, TRAINING_LABEL_NEGATIVE 
from learning import load_training_data

NEG_LABEL = -1
POS_LABEL = 1

class LinkSVM:

    def __init__(self, gpu_num, gpu_mem_frac=.95):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, device_count={'GPU': 0, 'CPU': 2}))

        self._create_architecture(gpu_num)

    def _create_architecture(self, gpu_num):
        with tf.device("/gpu:{}".format(gpu_num)):
            self.t_inputs = tf.placeholder(tf.float32, shape=[None, INPUT_VECTOR_SIZE])
            self.t_labels = tf.placeholder(tf.float32, shape=[None, 1])
            
            self.t_weights = tf.Variable(tf.random_normal(shape=[INPUT_VECTOR_SIZE, 1]))
            self.t_bias = tf.Variable(tf.random_normal(shape=[1, 1]))

            self.t_outputs = tf.matmul(self.t_inputs, self.t_weights) - self.t_bias

            self.t_alpha = 0.01

            self.loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - tf.multiply(self.t_outputs, self.t_labels))) + (self.t_alpha * tf.reduce_sum(tf.square(self.t_weights)))

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
                    self.t_inputs : batch_vecs,
                    self.t_labels : np.array(batch_labels, dtype=np.float32).reshape(-1, 1)
                }

                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

            print(loss)

    def evaluate(self, input_item):
        """
        Parameters
        --------------
        inputs : [np.ndarray]
            A list of float vectors of length 2048,
            represented as numpy arrays
        """
        
        feed_dict = {
            self.t_inputs : [input_item]
        }

        outputs = self.sess.run(self.t_outputs, feed_dict=feed_dict)

        assert len(outputs) == 1

        return np.sign(outputs[0])

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

    pos_max = 5000
    neg_max = 6000

    training_data = load_training_data(args.training_path, pos_max, neg_max)
    validation_data = dict(training_data)
    training_data[TRAINING_LABEL_POSITIVE] = training_data[TRAINING_LABEL_POSITIVE][:1200]
    training_data[TRAINING_LABEL_NEGATIVE] = training_data[TRAINING_LABEL_NEGATIVE][:1500]
    validation_data[TRAINING_LABEL_POSITIVE] = validation_data[TRAINING_LABEL_POSITIVE][1200:] 
    validation_data[TRAINING_LABEL_NEGATIVE] = validation_data[TRAINING_LABEL_NEGATIVE][1500:] 

    svm = LinkSVM(gpu_num=0)
    svm.train(training_data, 30, 100)

    pos_correct = 0
    neg_correct = 0
    for idx in range(3000):
        pos_item = validation_data[TRAINING_LABEL_POSITIVE][idx]
        neg_item = validation_data[TRAINING_LABEL_NEGATIVE][idx]

        pos_predict = svm.evaluate(pos_item)
        neg_predict = svm.evaluate(neg_item)

        if pos_predict == POS_LABEL:
            pos_correct += 1

        if neg_predict == NEG_LABEL:
            neg_correct += 1

    print(float(neg_correct + pos_correct) / 6000)
    print(float(neg_correct) / 3000)
    print(float(pos_correct) / 3000)
