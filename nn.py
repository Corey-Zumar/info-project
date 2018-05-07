import argparse
import sys
import os
import math
import numpy as np
import tensorflow as tf

from learning import INPUT_VECTOR_SIZE, NUM_CLASSES, TRAINING_LABEL_POSITIVE, TRAINING_LABEL_NEGATIVE 
from learning import load_training_data

NEG_TRAINING_LABEL = np.array([0,1], dtype=np.float32)
POS_TRAINING_LABEL = np.array([1,0], dtype=np.float32)

NEG_EVAL_LABEL = 0
POS_EVAL_LABEL = 1

class LinkNet:

    eval_net = None

    def __init__(self, gpu_num, weights_path=None, gpu_mem_frac=.95):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self._create_architecture(gpu_num, weights_path)

    def _create_architecture(self, gpu_num, weights_path=None):
        with tf.device("/gpu:{}".format(gpu_num)):
            self.inputs = tf.placeholder(tf.float32, shape=[None, INPUT_VECTOR_SIZE])
            self.labels = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

            fc1 = tf.contrib.layers.fully_connected(self.inputs, 256)
            fc2 = tf.contrib.layers.fully_connected(fc1, 64)
            fc3 = tf.contrib.layers.fully_connected(fc2, NUM_CLASSES)

            self.outputs = tf.nn.softmax(fc3)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.labels))

            if weights_path:
                saver = tf.train.Saver()
                saver.restore(self.sess, weights_path)

    def evaluate(self, input_item):
        feed_dict = {
            self.inputs : [input_item]
        }

        outputs = self.sess.run(self.outputs, feed_dict=feed_dict)
        assert len(outputs) == 1

        if outputs[0][0] > outputs[0][1]:
            return POS_EVAL_LABEL 
        else:
            return NEG_EVAL_LABEL 

        return outputs[0]

    def train(self, training_data, batch_size, num_epochs, learning_rate=.001):
        pos_vecs = training_data[TRAINING_LABEL_POSITIVE] 
        neg_vecs = training_data[TRAINING_LABEL_NEGATIVE]

        feature_vecs = np.array(pos_vecs + neg_vecs, dtype=np.float32)
        labels = np.array([POS_TRAINING_LABEL for _ in pos_vecs] + [NEG_TRAINING_LABEL for _ in neg_vecs], dtype=np.float32)
        
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

    def save(self, output_path):
        saver = tf.train.Saver()
        saver.save(self.sess, output_path)

def test_training(training_path):
    pos_max = 5000
    neg_max = 6000

    training_data = load_training_data(args.training_path, pos_max, neg_max)
        
    validation_data = dict(training_data)
    training_data[TRAINING_LABEL_POSITIVE] = training_data[TRAINING_LABEL_POSITIVE][:1200]
    training_data[TRAINING_LABEL_NEGATIVE] = training_data[TRAINING_LABEL_NEGATIVE][:2400]
    validation_data[TRAINING_LABEL_POSITIVE] = validation_data[TRAINING_LABEL_POSITIVE][1200:] 
    validation_data[TRAINING_LABEL_NEGATIVE] = validation_data[TRAINING_LABEL_NEGATIVE][2400:] 

    net = LinkNet(gpu_num=0)
    net.train(training_data, 30, 300)

    pos_correct = 0
    neg_correct = 0
    for idx in range(2000):
        pos_item = validation_data[TRAINING_LABEL_POSITIVE][idx]
        neg_item = validation_data[TRAINING_LABEL_NEGATIVE][idx]

        pos_predict = net.evaluate(pos_item)
        neg_predict = net.evaluate(neg_item)

        if pos_predict == POS_EVAL_LABEL:
            pos_correct += 1

        if neg_predict == NEG_EVAL_LABEL:
            neg_correct += 1

    print(float(neg_correct + pos_correct) / 4000)
    print(float(neg_correct) / 2000)
    print(float(pos_correct) / 2000)

    net.save("/home/ubuntu/info-project/nn_weights/nn_weights.ckpt")

def test_loading(weights_path, training_path):
    net = LinkNet(gpu_num=0, weights_path=weights_path)

    pos_max = 5000
    neg_max = 6000
    training_data = load_training_data(args.training_path, pos_max, neg_max)
    validation_data = dict(training_data)
    validation_data[TRAINING_LABEL_POSITIVE] = validation_data[TRAINING_LABEL_POSITIVE][1200:] 
    validation_data[TRAINING_LABEL_NEGATIVE] = validation_data[TRAINING_LABEL_NEGATIVE][1500:] 

    pos_correct = 0
    neg_correct = 0
    for idx in range(3000):
        pos_item = validation_data[TRAINING_LABEL_POSITIVE][idx]
        neg_item = validation_data[TRAINING_LABEL_NEGATIVE][idx]

        pos_predict = net.evaluate(pos_item)
        neg_predict = net.evaluate(neg_item)

        if pos_predict == POS_EVAL_LABEL:
            pos_correct += 1

        if neg_predict == NEG_EVAL_LABEL:
            neg_correct += 1

    print(float(neg_correct + pos_correct) / 6000)
    print(float(neg_correct) / 3000)
    print(float(pos_correct) / 3000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LinkNet')
    parser.add_argument('-t', '--training_path', type=str, help="The path to a directory containing model training data")
    parser.add_argument('-w', '--weights_path', type=str, help="The path to a directory containing saved model weights")

    args = parser.parse_args()

    if args.weights_path:
        test_loading(args.weights_path, args.training_path)
    else: 
        test_training(args.training_path)
