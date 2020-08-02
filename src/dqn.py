import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
MAX_POOL_NUM = 10000


class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('eval'):
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            pi_net = tflearn.fully_connected(
                merge_net, FEATURE_NUM, activation='relu')

            value = tflearn.fully_connected(pi_net, self.a_dim, activation='linear') 
            
            # wow, softmax!
            pi = tf.nn.softmax(value)

            return pi, value
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.pi, self.val = self.CreateNetwork(inputs=self.inputs)
        # self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.pool = []
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.loss = tflearn.mean_square(
            tf.reduce_sum(tf.multiply(self.val, self.acts), reduction_indices=1, keepdims=True),
            self.R)
        
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

    def predict(self, input):
        action = self.sess.run(self.val, feed_dict={
            self.inputs: input
        })
        return action[0]

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        for (s,a,v) in zip(s_batch, a_batch, v_batch):
            self.pool.append([s, a, v])
            if len(self.pool) > MAX_POOL_NUM:
                pop_item = np.random.randint(len(self.pool))
                self.pool.pop(pop_item)
        if len(self.pool) > 4096:
            s_batch, a_batch, v_batch = [], [], []
            for p in range(512):
                pop_item = np.random.randint(len(self.pool))
                s_, a_, v_ = self.pool[pop_item]
                s_batch.append(s_)
                a_batch.append(a_)
                v_batch.append(v_)

            self.sess.run(self.val_opt, feed_dict={
                self.inputs: s_batch,
                self.acts: a_batch,
                self.R: v_batch
            })

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:    
            v_batch = self.sess.run(self.val, feed_dict={
                self.inputs: s_batch
            })
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)
