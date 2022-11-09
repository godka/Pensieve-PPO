import math
import numpy as np
import tensorflow.compat.v1 as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-4
PAST_LEN = 5
A_SAT = 2
GAMMA = 0.99
# PPO2
EPS = 0.2
DIM_SIZE = 1
ENTROPY_WEIGHT = 0.1
MAX_SAT = 5

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.a_dim], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')
            split_6 = tflearn.conv_1d(inputs[:, 6:7, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_7 = tflearn.conv_1d(inputs[:, 7:8, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_8 = tflearn.conv_1d(inputs[:, 8:9, :A_SAT], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_9 = tflearn.conv_1d(inputs[:, 9:10, :A_SAT], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_10 = tflearn.conv_1d(inputs[:, 10:11, :MAX_SAT - A_SAT], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_11 = tflearn.conv_1d(inputs[:, 11:12, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_12 = tflearn.conv_1d(inputs[:, 12:13, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_13 = tflearn.conv_1d(inputs[:, 13:14, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_14 = tflearn.conv_1d(inputs[:, 14:15, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_15 = tflearn.conv_1d(inputs[:, 15:16, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_6_flat = tflearn.flatten(split_6)
            split_7_flat = tflearn.flatten(split_7)
            split_8_flat = tflearn.flatten(split_8)
            split_9_flat = tflearn.flatten(split_9)
            split_10_flat = tflearn.flatten(split_10)
            split_11_flat = tflearn.flatten(split_11)
            split_12_flat = tflearn.flatten(split_12)
            split_13_flat = tflearn.flatten(split_13)
            split_14_flat = tflearn.flatten(split_14)
            split_15_flat = tflearn.flatten(split_15)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5, split_6_flat,
                 split_7_flat, split_8_flat, split_9_flat, split_10_flat, split_11_flat, split_12_flat,
                 split_13_flat, split_14_flat, split_15_flat], 'concat')

            pi_net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            pi = tflearn.fully_connected(pi_net, self.a_dim, activation='softmax')

        with tf.variable_scope('critic'):
            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.a_dim], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')
            split_6 = tflearn.conv_1d(inputs[:, 6:7, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_7 = tflearn.conv_1d(inputs[:, 7:8, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_8 = tflearn.conv_1d(inputs[:, 8:9, :A_SAT], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_9 = tflearn.conv_1d(inputs[:, 9:10, :A_SAT], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_10 = tflearn.conv_1d(inputs[:, 10:11, :MAX_SAT - A_SAT], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_11 = tflearn.conv_1d(inputs[:, 11:12, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_12 = tflearn.conv_1d(inputs[:, 12:13, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_13 = tflearn.conv_1d(inputs[:, 13:14, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_14 = tflearn.conv_1d(inputs[:, 14:15, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')
            split_15 = tflearn.conv_1d(inputs[:, 15:16, :PAST_LEN], FEATURE_NUM, DIM_SIZE, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_6_flat = tflearn.flatten(split_6)
            split_7_flat = tflearn.flatten(split_7)
            split_8_flat = tflearn.flatten(split_8)
            split_9_flat = tflearn.flatten(split_9)
            split_10_flat = tflearn.flatten(split_10)
            split_11_flat = tflearn.flatten(split_11)
            split_12_flat = tflearn.flatten(split_12)
            split_13_flat = tflearn.flatten(split_13)
            split_14_flat = tflearn.flatten(split_14)
            split_15_flat = tflearn.flatten(split_15)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5, split_6_flat,
                 split_7_flat, split_8_flat, split_9_flat, split_10_flat, split_11_flat, split_12_flat,
                 split_13_flat, split_14_flat, split_15_flat], 'concat')

            value_net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            value = tflearn.fully_connected(value_net, 1, activation='linear')
            return pi, value
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def r(self, pi_new, pi_old, acts):
        return tf.reduce_sum(tf.multiply(pi_new, acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(pi_old, acts), reduction_indices=1, keepdims=True)

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self._entropy_weight = np.log(self.a_dim)
        self.H_target = 0.1

        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)
        self.pi, self.val = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        
        self.entropy = -tf.reduce_sum(tf.multiply(self.real_out, tf.log(self.real_out)), reduction_indices=1, keepdims=True)
        self.adv = tf.stop_gradient(self.R - self.val)
        self.ppo2loss = tf.minimum(self.r(self.real_out, self.old_pi, self.acts) * self.adv, 
                            tf.clip_by_value(self.r(self.real_out, self.old_pi, self.acts), 1 - EPS, 1 + EPS) * self.adv
                        )
        self.dual_loss = tf.where(tf.less(self.adv, 0.), tf.maximum(self.ppo2loss, 3. * self.adv), self.ppo2loss)

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.network_params += \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.policy_loss = - tf.reduce_sum(self.dual_loss) - self.entropy_weight * tf.reduce_sum(self.entropy)
        self.policy_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.policy_loss)
        self.val_loss = tflearn.mean_square(self.val, self.R)
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate * 10.).minimize(self.val_loss)

    def predict(self, input):
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: input
        })
        return action[0]
    
    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        self.sess.run([self.policy_opt, self.val_opt], feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.R: v_batch, 
            self.old_pi: p_batch,
            self.entropy_weight: self._entropy_weight
        })
        # adaptive entropy weight
        # https://arxiv.org/abs/2003.13590
        p_batch = np.clip(p_batch, ACTION_EPS, 1. - ACTION_EPS)
        _H = np.mean(np.sum(-np.log(p_batch) * p_batch, axis=1))
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * ENTROPY_WEIGHT

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
