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
# PPO2
EPS = 0.2

class Network():
    def sample_gumbel(self, shape, eps=1e-20):
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax(self, logits):
        gumbel_softmax_sample = tf.log(logits) + self.sample_gumbel(tf.shape(logits))
        # print(gumbel_softmax_sample.get_shape().as_list())
        return tf.argmax(gumbel_softmax_sample, axis=1)

    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
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
            value_net = tflearn.fully_connected(
                merge_net, FEATURE_NUM, activation='relu')

            pi = tflearn.fully_connected(pi_net, self.a_dim, activation='softmax')

            bit_rate = self.gumbel_softmax(pi)
            # log_pi = tf.nn.sparse_softmax_cross_entropy_with_logits(real_out, bit_rate)

            value = tflearn.fully_connected(value_net, 1, activation='linear')

            return pi, value, bit_rate
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def r(self, pi_new, pi_old, acts):
        ratio = tf.reduce_sum(pi_new * tf.one_hot(tf.squeeze(acts), self.a_dim), axis=1) \
            / (pi_old + ACTION_EPS)
        return ratio

    def sample(self, logits):
        noise = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits - tf.log(-tf.log(noise)), 1)
    
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self._entropy = 5.
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        
        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.old_log_pi = tf.placeholder(tf.float32, [None, 1])
        self.acts = tf.placeholder(tf.int32, [None, 1])

        self.entropy_weight = tf.placeholder(tf.float32)
        self.pi, self.val, self.action = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)

        self.entropy = tf.multiply(self.real_out, tf.log(self.real_out))
        self.adv = tf.stop_gradient(self.R - self.val)

        self.ppo2loss = tf.minimum(self.r(self.real_out, self.old_log_pi, self.acts) * self.adv, 
                            tf.clip_by_value(self.r(self.real_out, self.old_log_pi, self.acts), 1 - EPS, 1 + EPS) * self.adv
                        )
        # https://arxiv.org/pdf/1912.09729.pdf
        self.dualppo = tf.cast(tf.less(self.adv, 0.), dtype=tf.float32)  * \
            tf.maximum(self.ppo2loss, 3. * self.adv) + \
            tf.cast(tf.greater_equal(self.adv, 0.), dtype=tf.float32) * \
            self.ppo2loss
        
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.policy_loss = - tf.reduce_sum(self.dualppo) \
            + self.entropy_weight * tf.reduce_sum(self.entropy)
        self.policy_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.policy_loss)

        self.val_loss = tflearn.mean_square(self.val, self.R)
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate * 10.).minimize(self.val_loss)

    def predict(self, input):
        prob, action = self.sess.run([self.real_out, self.action], feed_dict={
            self.inputs: input
        })
        action_prob = prob[0, action[0]]
        return action[0], action_prob, prob[0]

    def set_entropy_decay(self, decay=0.6):
        self._entropy *= decay

    def get_entropy(self, step):
        return np.clip(self._entropy, 0.01, 5.)

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        s_batch, a_batch, p_batch, v_batch = \
            tflearn.data_utils.shuffle(s_batch, a_batch, p_batch, v_batch)
        self.sess.run([self.policy_opt, self.val_opt], feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.R: v_batch, 
            self.old_log_pi: p_batch,
            self.entropy_weight: self.get_entropy(epoch)
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
