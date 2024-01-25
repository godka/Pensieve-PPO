import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
EPS = 0.2  # PPO2 epsilon

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # Actor network
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.a_dim, FEATURE_NUM)
        self.fc3_actor = nn.Linear(1, FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM * self.s_dim[0], FEATURE_NUM)
        self.pi_head = nn.Linear(FEATURE_NUM, action_dim)

    def forward(self, inputs):
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, FEATURE_NUM))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        pi_net = F.relu(self.fc4_actor(merge_net))
        pi = F.softmax(self.pi_head(pi_net), dim=-1)
        pi = torch.clamp(pi, ACTION_EPS, 1. - ACTION_EPS)
        return pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic network
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.a_dim, FEATURE_NUM)
        self.fc3_actor = nn.Linear(1, FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM * self.s_dim[0], FEATURE_NUM)
        self.val_head = nn.Linear(FEATURE_NUM, 1)

    def forward(self, inputs):
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, FEATURE_NUM))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        value_net = F.relu(self.fc4_actor(merge_net))
        value = self.val_head(value_net)
        return value
    
class Network():
    def __init__(self, state_dim, action_dim, learning_rate):

        self.s_dim = state_dim
        self.action_dim = action_dim
        self._entropy_weight = np.log(action_dim)
        self.H_target = 0.1
        self.PPO_TRAINING_EPO = 5

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.lr_rate = learning_rate
        self.optimizer = optim.Adam(list(self.actor.parameters()) + \
                                    list(self.critic.parameters()), lr=learning_rate)

    def get_network_params(self):
        return [self.actor.state_dict(), self.critic.state_dict()]
    
    def set_network_params(self, input_network_params):
        actor_net_params, critic_net_params = input_network_params
        self.actor.load_state_dict(actor_net_params)
        self.critic.load_state_dict(critic_net_params)

    def r(self, pi_new, pi_old, acts):
        return torch.sum(pi_new * acts, dim=1, keepdim=True) / \
               torch.sum(pi_old * acts, dim=1, keepdim=True)

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        s_batch = torch.from_numpy(s_batch).to(torch.float32)
        a_batch = torch.from_numpy(a_batch).to(torch.float32)
        p_batch = torch.from_numpy(p_batch).to(torch.float32)
        v_batch = torch.from_numpy(v_batch).to(torch.float32)

        for _ in range(self.PPO_TRAINING_EPO):
            pi = self.actor.forward(s_batch)
            val = self.critic.forward(s_batch)

            # loss
            adv = v_batch - val.detach()
            ratio = self.r(pi, p_batch, a_batch)
            ppo2loss = torch.min(ratio * adv, torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv)
            # Dual-PPO
            dual_loss = torch.where(adv < 0, torch.max(ppo2loss, 3. * adv), ppo2loss)
            loss_entropy = torch.sum(-pi * torch.log(pi), dim=1, keepdim=True)

            loss = -dual_loss.mean() + 10. * F.mse_loss(val, v_batch) - self._entropy_weight * loss_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update entropy weight
        _H = (-(torch.log(p_batch) * p_batch).sum(dim=1)).mean().item()
        _g = _H - self.H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1 * self.PPO_TRAINING_EPO
        self._entropy_weight = max(self._entropy_weight, 1e-2)

    def predict(self, input):
        with torch.no_grad():
            input = torch.from_numpy(input).to(torch.float32)
            pi = self.actor.forward(input)[0]
            return pi.numpy()

    def load_model(self, nn_model):
        actor_model_params, critic_model_params = torch.load(nn_model)
        self.actor.load_state_dict(actor_model_params)
        self.critic.load_state_dict(critic_model_params)

    def save_model(self, nn_model):
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        torch.save(model_params, nn_model)

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        R_batch = np.zeros_like(r_batch)

        if terminal:
            # in this case, the terminal reward will be assigned as r_batch[-1]
            R_batch[-1] = r_batch[-1]  # terminal state
        else:
            val = self.critic.forward(s_batch)
            R_batch[-1] = val[-1]  # bootstrap from last state

        for t in reversed(range(len(r_batch) - 1)):
            R_batch[t] = r_batch[t] + GAMMA * R_batch[t + 1]

        return list(R_batch)
           
