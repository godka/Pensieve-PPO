import math
import numpy as np
import tensorflow.compat.v1 as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
MAX_POOL_NUM = 500000
TAU = 1e-5

class Network():
    def CreateTarget(self, inputs):
        with tf.variable_scope('target'):
            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], FEATURE_NUM, 1, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, 1, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 1, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')
            net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            value = tflearn.fully_connected(net, self.a_dim, activation='linear') 
            
            return value

    def CreateEval(self, inputs):
        with tf.variable_scope('eval', reuse=tf.AUTO_REUSE):
            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], FEATURE_NUM, 1, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, 1, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 1, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')
            net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            value = tflearn.fully_connected(net, self.a_dim, activation='linear')
            
            return value
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def update(self, tau = 1e-4):
        eval_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval')
        target_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        for (e_param, t_param) in zip(eval_params, target_params):
            t_param.assign((1 - tau) * t_param + tau * e_param)
        
    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.done = tf.placeholder(tf.float32, [None, 1])

        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.ns_inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.eval = self.CreateEval(inputs=self.inputs)
        self.eval_ns = self.CreateEval(inputs=self.ns_inputs)
        self.target = self.CreateTarget(inputs=self.ns_inputs)
        self.max_target = tf.reduce_max(self.target, axis=-1)
        self.double_target = tf.reduce_sum(tf.multiply(self.target, \
                                tf.one_hot(tf.argmax(self.eval_ns, axis=-1), self.a_dim)), reduction_indices=1, keepdims=True)

        self.R = tf.stop_gradient(self.r + GAMMA * (1 - self.done) * self.double_target)
        self.pool = []

        self.eval_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval')
        self.target_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        self.soft_update = [tf.assign(ta, (1 - TAU) * ta + TAU * ea)
                for ta, ea in zip(self.target_params, self.eval_params)]
        
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval')
        self.network_params += \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

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
            tf.reduce_sum(tf.multiply(self.eval, self.acts), reduction_indices=1, keepdims=True),
            self.R)
        
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

    def predict(self, input):
        action = self.sess.run(self.eval, feed_dict={
            self.inputs: input
        })
        return action[0]

    def train(self, s_batch, a_batch, p_batch, r_batch, d_batch, epoch):
        # ns: next state
        for (s, a, v, ns, d) in zip(s_batch, a_batch, r_batch, p_batch, d_batch):
            if len(self.pool) > MAX_POOL_NUM:
                pop_item = np.random.randint(len(self.pool))
                self.pool[pop_item] = [s, a, v, ns, d]
            else:
                self.pool.append([s, a, v, ns, d])
        
        if len(self.pool) > 4096:
            s_batch, a_batch, r_batch, ns_batch, d_batch = [], [], [], [], []

            #for _ in range(512):
            pop_items = np.random.randint(len(self.pool), size=1024)
            for pop_item in pop_items:
                s_, a_, r_, ns_, d_ = self.pool[pop_item]
                s_batch.append(s_)
                a_batch.append(a_)
                r_batch.append(r_)
                ns_batch.append(ns_)
                d_batch.append(d_)

            self.sess.run(self.val_opt, feed_dict={
                self.inputs: s_batch,
                self.acts: a_batch,
                self.r: r_batch,
                self.ns_inputs: ns_batch,
                self.done: d_batch
            })
            self.sess.run(self.soft_update)
