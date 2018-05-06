import argparse
import sys
import os
import math
import numpy as np
import tensorflow as tf

from learning import INPUT_VECTOR_SIZE, NUM_CLASSES, TRAINING_LABEL_POSITIVE, TRAINING_LABEL_NEGATIVE 
from learning import load_training_data

NEG_LABEL = np.array([0,1], dtype=np.float32)
POS_LABEL = np.array([1,0], dtype=np.float32)

class LinkNet:

    def __init__(self, gpu_num, gpu_mem_frac=.95):
        self._create_architecture(gpu_num)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    def _create_architecture(self, gpu_num):
        with tf.device("/gpu:{}".format(gpu_num)):
            self.inputs = tf.placeholder(tf.float32, shape=[None, INPUT_VECTOR_SIZE])
            self.labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

            fc1 = tf.contrib.layers.fully_connected(self.inputs, 256)
            fc2 = tf.contrib.layers.fully_connected(fc1, 64)
            fc3 = tf.contrib.layers.fully_connected(fc2, NUM_CLASSES)

            self.outputs = tf.nn.softmax(fc3)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.labels))

    def evaluate(self, input_item):
        feed_dict = {
            self.inputs : [input_item]
        }

        outputs = self.sess.run(self.outputs, feed_dict=feed_dict)
        return outputs
    
    def train(self, training_data, batch_size, num_epochs, learning_rate=.001):
        pos_vecs = training_data[TRAINING_LABEL_POSITIVE][5:] 
        neg_vecs = training_data[TRAINING_LABEL_NEGATIVE][5:]

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

        for i in range(5):
            print("POS", self.evaluate(training_data[TRAINING_LABEL_POSITIVE][i]))
            print("NEG", self.evaluate(training_data[TRAINING_LABEL_NEGATIVE][i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LinkNet')
    parser.add_argument('-t', '--training_path', type=str, help="The path to a directory containing model training data")

    args = parser.parse_args()

    pos_max = 1000
    neg_max = 1500

    training_data = load_training_data(args.training_path, pos_max, neg_max)
    net = LinkNet(gpu_num=0)
    net.train(training_data, 30, 100)
    print(net.evaluate(training_data[TRAINING_LABEL_POSITIVE][0]))
    print(net.evaluate(training_data[TRAINING_LABEL_NEGATIVE][0]))
