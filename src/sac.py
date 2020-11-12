import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.9
# PPO2
EPS = 0.2


def CreateCriticNetwork(inputs, a_dim, name):
    with tf.variable_scope('Q_network' + name):
        split_0 = tflearn.fully_connected(
            inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
        split_1 = tflearn.fully_connected(
            inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
        split_2 = tflearn.conv_1d(
            inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
        split_3 = tflearn.conv_1d(
            inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
        split_4 = tflearn.conv_1d(
            inputs[:, 4:5, :a_dim], FEATURE_NUM, 4, activation='relu')
        split_5 = tflearn.fully_connected(
            inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

        split_2_flat = tflearn.flatten(split_2)
        split_3_flat = tflearn.flatten(split_3)
        split_4_flat = tflearn.flatten(split_4)

        merge_net = tflearn.merge(
            [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

        q_net = tflearn.fully_connected(
            merge_net, FEATURE_NUM, activation='relu')
        q = tflearn.fully_connected(q_net, a_dim, activation='linear') 
        return q
    
def CreateActorNetwork(inputs, a_dim):
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
            inputs[:, 4:5, :a_dim], FEATURE_NUM, 4, activation='relu')
        split_5 = tflearn.fully_connected(
            inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

        split_2_flat = tflearn.flatten(split_2)
        split_3_flat = tflearn.flatten(split_3)
        split_4_flat = tflearn.flatten(split_4)

        merge_net = tflearn.merge(
            [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

        pi_net = tflearn.fully_connected(
            merge_net, FEATURE_NUM, activation='relu')
        pi = tflearn.fully_connected(pi_net, a_dim, activation='softmax') 
        return pi

class Network():
    def __init__(self, sess, state_dim, action_dim, learning_rate, name):
        self.name = name
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.q_target = tf.placeholder(tf.float32, [None, 1])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)
        with tf.variable_scope(self.name):
            self.q_net1 = CreateCriticNetwork(self.inputs, a_dim=self.a_dim, name= '1')
            self.q_net2 = CreateCriticNetwork(self.inputs, a_dim=self.a_dim, name= '2')
            
            self.actor_net_2 = CreateActorNetwork(self.inputs, a_dim=self.a_dim)
            self.actor_net = tf.clip_by_value(self.actor_net_2, 1e-4, 1. - 1e-4)
        
        self.actor_network_params = \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.name + '/actor')
        self.q1_network_params = \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.name + '/Q_network1')
            
        self.q2_network_params = \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= self.name + '/Q_network2')
        
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        self.input_network_params_q1 = []
        self.input_network_params_q2 = []
        self.input_network_params_actor = []
        for param in self.q1_network_params:
            self.input_network_params_q1.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        for param in self.q2_network_params:
            self.input_network_params_q2.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        for param in self.actor_network_params:
            self.input_network_params_actor.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        
        self.set_network_params_op_q1, self.set_network_params_op_q2, self.set_network_params_op_actor  = [], [], []
        for idx, param in enumerate(self.input_network_params_q1):
            self.set_network_params_op_q1.append(
                self.q1_network_params[idx].assign(param))
        for idx, param in enumerate(self.input_network_params_q2):
            self.set_network_params_op_q2.append(
                self.q2_network_params[idx].assign(param))
        for idx, param in enumerate(self.input_network_params_actor):
            self.set_network_params_op_actor.append(
                self.actor_network_params[idx].assign(param))
        
        self.q1a_val = tf.reduce_sum(tf.multiply(self.q_net1, self.acts), reduction_indices=1, keepdims=True)
        self.q2a_val = tf.reduce_sum(tf.multiply(self.q_net2, self.acts), reduction_indices=1, keepdims=True)
        self.loss_q1 = tflearn.mean_square(self.q1a_val, self.q_target)
        self.loss_q2 = tflearn.mean_square(self.q2a_val, self.q_target)
        self.q_loss = self.loss_q1 + self.loss_q2
        
        self.log_pi_prob = tf.log(self.actor_net)
        self.entropy = -tf.multiply(self.actor_net, tf.log(self.actor_net))
        self.min_q = tf.minimum(self.q_net1, self.q_net2)
        self.pi_loss = -tf.reduce_sum(self.entropy_weight*self.entropy + tf.multiply(tf.stop_gradient(self.min_q), self.actor_net))
        
        self.pi_optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.pi_loss)
        self.q_optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.q_loss)
        
        
    def get_entropy_weight(self, step):
        # max_lr = 0.5
        # min_lr = 0.05
        # return np.maximum(min_lr, min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(step * np.pi / 100000)))
        return np.clip(6 - step / 10000, 0.01, 6)
        # return 5
        # return 0.2
        # if step < 2000:
        #     return 5.
        # elif step < 5000:
        #     return 3.
        # elif step < 10000:
        #     return 1.
        # else:
        #     return np.clip(1. - step / 50000, 0.1, 1.)

    def get_q1a(self, input, acts):
        q1a = self.sess.run(self.q1a_val, feed_dict={
            self.inputs: input,
            self.acts: acts
        })
        return q1a
    
    def get_q2a(self, input, acts):
        q2a = self.sess.run(self.q2a_val, feed_dict={
            self.inputs: input,
            self.acts: acts
        })
        return q2a
    def get_action_prob(self, input):
        action = self.sess.run(self.actor_net, feed_dict={
            self.inputs: input
        })
        return action[0]
    
    def get_network_params(self):
        return [self.get_network_params_q1(), self.get_network_params_q2(), self.get_network_params_actor()]
    
    def set_network_params(self, input_params):
        input_q1, input_q2, input_actor = input_params
        
        self.set_network_params_q1(input_q1)
        self.set_network_params_q2(input_q2)
        self.set_network_params_actor(input_actor)

    

    def get_network_params_q1(self):
        return self.sess.run(self.q1_network_params)

    def set_network_params_q1(self, input_network_params):
        self.sess.run(self.set_network_params_op_q1, feed_dict={
            i: d for i, d in zip(self.input_network_params_q1, input_network_params)
        })

    def get_network_params_q2(self):
        return self.sess.run(self.q2_network_params)

    def set_network_params_q2(self, input_network_params):
        self.sess.run(self.set_network_params_op_q2, feed_dict={
            i: d for i, d in zip(self.input_network_params_q2, input_network_params)
        })
    
    def get_network_params_actor(self):
        return self.sess.run(self.actor_network_params)

    def set_network_params_actor(self, input_network_params):
        self.sess.run(self.set_network_params_op_actor, feed_dict={
            i: d for i, d in zip(self.input_network_params_actor, input_network_params)
        })
    
    def train(self, s_batch, a_batch, q_target, step):
        
        pi_loss = self.sess.run(self.pi_loss, feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.q_target: q_target,
            self.entropy_weight: self.get_entropy_weight(step)
        })

        self.sess.run([self.pi_optimize, self.q_optimize], feed_dict={
            self.inputs: s_batch,
            self.acts:a_batch,
            self.q_target:q_target,
            self.entropy_weight: self.get_entropy_weight(step)
        })
        
        return pi_loss
    
    def get_entropy(self, s_batch):
        return self.sess.run(self.entropy, feed_dict={
            self.inputs: s_batch
        })

    def get_min_q(self, s_batch):
        return self.sess.run(self.min_q, feed_dict={
            self.inputs:s_batch
        })

    '''Here s_batch is st+1'''
    def compute_v(self, s_batch, r_batch, dones, eval_entropy):
        assert len(s_batch) == len(r_batch)
        V_batch = self.sess.run(self.min_q, feed_dict={
            self.inputs:s_batch
        })
        pi_batch, entropy = self.sess.run([self.actor_net, self.entropy], feed_dict={
            self.inputs:s_batch
        })

        R_batch = r_batch + GAMMA * (1-dones) * np.add.reduce(pi_batch*V_batch + entropy, axis=1)
        R_batch = R_batch[..., np.newaxis]

        return R_batch
        