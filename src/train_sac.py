import multiprocessing as mp
from os import name
import numpy as np
import logging
import os
import sys
from abr import ABREnv
# import ppo2 as network
import tensorflow as tf
from sac import Network
import ReplayBuffer

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

POLYAK = 0.995
S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE =1e-4
CRITIC_LR_RATE = 1e-3
NUM_AGENTS = 6
TRAIN_SEQ_LEN = 300  # take as a train batch
TRAIN_EPOCH = 1000000
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './results'
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './results/log'
PPO_TRAINING_EPO = 5
SAMPLE_SZIE = 1024
# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python rl_test_sac.py ' + nn_model)

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)

def central_agent(net_params_queues, exp_queues):
    replay_buffer = ReplayBuffer.ReplayBuffer(obs_dim=S_DIM, act_dim=A_DIM, size=100000)
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    tf_config=tf.ConfigProto(intra_op_parallelism_threads=5,
                            inter_op_parallelism_threads=5)
    with tf.Session(config = tf_config) as sess, open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        summary_ops, summary_vars = build_summaries()

        # actor = network.Network(sess,
        #         state_dim=S_DIM, action_dim=A_DIM,
        #         learning_rate=ACTOR_LR_RATE)

        target_net = Network(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE, name='target')
        eval_net = Network(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE, name='eval')

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=10)  # save neural net parameters

        # restore neural net parameters
        # nn_model = NN_MODEL
        # if nn_model is not None:  # nn_model is the path to file
        #     saver.restore(sess, nn_model)
        #     print("Model restored.")

        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(1, TRAIN_EPOCH):
            if epoch % 10 == 0:
                print(epoch, replay_buffer.size)
            # synchronize the network parameters of work agent
            network_params = eval_net.get_network_params()
            # print(network_params[0].keys())
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(network_params)

            
            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, done_batch, entropy_batch = exp_queues[i].get()
                for s, a, r, s_, done in zip(s_batch[:-1], a_batch, r_batch, s_batch[1:], done_batch):
                    replay_buffer.store(s, a, r, s_, done)
            
            sample_data = replay_buffer.sample_batch(SAMPLE_SZIE)
            
            '''Training logic'''

            obs, a, r, obs2, done = sample_data['obs1'], sample_data['acts'], sample_data['rews'], sample_data['obs2'], sample_data['done']

            eval_entropy = eval_net.get_entropy(obs2)

            

            target_q = target_net.compute_v(obs2, r, done, eval_entropy)
            # print(target_q)
            pi_loss = eval_net.train(obs, a, target_q, epoch)

            # print(q_loss)
            new_params = eval_net.get_network_params()
            old_params = target_net.get_network_params()

            target_params_updates = []
            for new_param, old_param in zip(new_params[:-1], old_params[:-1]):
                target_params_update = []
                for new_p, old_p in zip(new_param, old_param):
                    target_params_update.append(old_p * POLYAK + (1-POLYAK)*new_p)
                target_params_updates.append(target_params_update)
            target_params_updates.append(new_params[-1])

            target_net.set_network_params(target_params_updates)
            
            # actor.train(s_batch, a_batch, v_batch, epoch)
            
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                print('model saved in ' + save_path)
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: pi_loss/SAMPLE_SZIE,
                    summary_vars[1]: avg_reward,
                    summary_vars[2]: avg_entropy
                })
                writer.add_summary(summary_str, epoch)
                writer.flush()

def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
        actor = Network(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE, name='hehe')

        # initial synchronization of the network parameters from the coordinator
        net_params = net_params_queue.get()
        actor.set_network_params(net_params)


        time_stamp = 0

        for epoch in range(TRAIN_EPOCH):
            obs = env.reset()
            s_batch, a_batch, r_batch, done_batch, entropy_batch = [], [], [], [], []
            for _ in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)

                action_prob = actor.get_action_prob(
                    np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
                
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(
                    1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                
                entropy = -np.dot(action_prob, np.log(action_prob))
                obs, rew, done, info = env.step(bit_rate)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)
                r_batch.append(rew)
                done_batch.append(done)
                entropy_batch.append(entropy)
                if done:
                    break
            # v_batch, td_target = actor.compute_v(s_batch, a_batch, r_batch, done)
            exp_queue.put([s_batch, a_batch, r_batch, done_batch, entropy_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("Beta", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", eps_total_reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)

    summary_vars = [td_loss, eps_total_reward, entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()