import multiprocessing as mp
import numpy as np
import os
from env.multi_bw_share.env import ABREnv
from models.rl_multi_bw_share.ppo_spec import ppo_implicit as network
import tensorflow.compat.v1 as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [6 + 3, 8]
A_DIM = 6
A_SAT = 2
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 5000000
MODEL_SAVE_INTERVAL = 1000
RANDOM_SEED = 42
SUMMARY_DIR = './ppo_imp'
MODEL_DIR = '..'
TRAIN_TRACES = '../../data/sat_data/train/'
TEST_LOG_FOLDER = './test_results_imp'
PPO_TRAINING_EPO = 5

import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--user', type=int, default=3)
args = parser.parse_args()
USERS = args.user
# A_SAT = USERS + 1

TEST_LOG_FOLDER += str(USERS) + '/'
SUMMARY_DIR += str(USERS)
LOG_FILE = SUMMARY_DIR + '/log'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    


def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    # os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    print('python test_implicit.py ' + nn_model + ' ' + str(USERS))
    os.system('python test_implicit.py ' + nn_model + ' ' + str(USERS))

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = [] # , # entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    # entropy.append(float(parse[-2]))
                    reward.append(float(parse[-6]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        # entropies.append(np.mean(entropy[1:]))

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

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    tf_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    with tf.Session(config = tf_config) as sess, open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        summary_ops, summary_vars = build_summaries()

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM * A_SAT,
                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        
        while True:  # assemble experiences from agents, compute the gradients
        # for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_ = exp_queues[i].get(timeout=10)
                s += s_
                a += a_
                p += p_
                g += g_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(g)

            # print(s_batch[0], a_batch[0], p_batch[0], v_batch[0], epoch)
            for _ in range(PPO_TRAINING_EPO):
                actor.train(s_batch, a_batch, p_batch, v_batch, None)
            
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: actor._entropy_weight,
                    summary_vars[1]: avg_reward,
                    summary_vars[2]: avg_entropy
                })
                writer.add_summary(summary_str, epoch)
                writer.flush()


def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id, num_agents=USERS)
    with tf.Session() as sess:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM * A_SAT,
                                learning_rate=ACTOR_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        time_stamp = 0

        for epoch in range(TRAIN_EPOCH):
            bit_rate = [0 for _ in range(USERS)]
            sat = [0 for _ in range(USERS)]
            action_prob = [[] for _ in range(USERS)]
            
            obs = env.reset()
            
            for agent in range(USERS):
                obs[agent] = env.reset_agent(agent)

                action_prob[agent] = actor.predict(
                    np.reshape(obs[agent], (1, S_DIM[0], S_DIM[1])))
            
                # gumbel noise
                noise = np.random.gumbel(size=len(action_prob[agent]))
                bit_rate[agent] = np.argmax(np.log(action_prob[agent]) + noise)

                sat[agent] = bit_rate[agent] // A_DIM
                
                env.set_sat(agent, sat[agent])
    
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            s_batch_user, a_batch_user, p_batch_user, r_batch_user = \
                [[]for _ in range(USERS)], [[]for _ in range(USERS)], \
                [[]for _ in range(USERS)], [[]for _ in range(USERS)]
            
            for step in range(TRAIN_SEQ_LEN):
                agent = env.get_first_agent()
                
                if agent == -1:
                    break

                s_batch_user[agent].append(obs[agent])
                    
                obs[agent], rew, done, info = env.step(bit_rate[agent], agent)

                action_vec = np.zeros(A_DIM * A_SAT)
                action_vec[bit_rate[agent]] = 1
                a_batch_user[agent].append(action_vec)
                r_batch_user[agent].append(rew)
                p_batch_user[agent].append(action_prob[agent])

                if not done:
                    
                    action_prob[agent] = actor.predict(
                        np.reshape(obs[agent], (1, S_DIM[0], S_DIM[1])))
                
                    # gumbel noise
                    noise = np.random.gumbel(size=len(action_prob[agent]))
                    bit_rate[agent] = np.argmax(np.log(action_prob[agent]) + noise)

                    sat[agent] = bit_rate[agent] // A_DIM
                    
                    env.set_sat(agent, sat[agent])
                    
                if env.check_end():
                    break
                
                # if agent_id == 0:
                #     print(ppo_spec.net_env.video_chunk_counter)
                #     print([len(batch_user) for batch_user in s_batch_user])
                #     print([len(batch_user) for batch_user in r_batch_user])
                    
            for batch_user in s_batch_user:
                s_batch += batch_user
            for batch_user in a_batch_user:
                a_batch += batch_user
            for batch_user in p_batch_user:
                p_batch += batch_user
            for batch_user in r_batch_user:
                r_batch += batch_user
                
            # if agent_id == 0:
            #     print(len(s_batch), len(a_batch), len(r_batch))
            v_batch = actor.compute_v(s_batch, a_batch, r_batch, env.check_end())
            exp_queue.put([s_batch, a_batch, p_batch, v_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)


def build_summaries():
    entropy_weight = tf.Variable(0.)
    tf.summary.scalar("Entropy Weight", entropy_weight)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", eps_total_reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)

    summary_vars = [entropy_weight, eps_total_reward, entropy]
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
