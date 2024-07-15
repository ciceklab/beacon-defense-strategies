import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.max_action = 2
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_dim),
                        # nn.Sigmoid(),
                        nn.Tanh()
                    )

    def forward(self, s):
        s = self.actor(s)
        a = self.max_action * s  # [-max,max]
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim + action_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def forward(self, s, a):
        c = self.critic(torch.cat([s, a], 1))
        return c
    

class DDPG(object):
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, K_epochs=300):
        self.K_epochs = K_epochs
        self.batch_size = 256  # batch size
        self.GAMMA = gamma # discount factor
        self.TAU = tau  # Softly update the target network
        self.lr = lr  # learning rate

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()

        self.buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim)
        self.actor_scheduler = torch.optim.lr_scheduler.CyclicLR(self.actor_optimizer, base_lr=0.00001, max_lr=0.001, cycle_momentum=False, mode='triangular2', step_size_up=7, step_size_down=5)
        self.critic_scheduler = torch.optim.lr_scheduler.CyclicLR(self.critic_optimizer, base_lr=0.00001, max_lr=0.001, cycle_momentum=False, mode='triangular2', step_size_up=7, step_size_down=5)

    def select_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.numpy().flatten()
        return a

    def update(self):
        for _ in range(self.K_epochs):
            batch_s, batch_a, batch_r, batch_s_, batch_dw = self.buffer.sample(self.batch_size)

            # Compute the target Q
            with torch.no_grad():  # target_Q has no gradient
                Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
                target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

            # Compute the current Q and the critic loss
            current_Q = self.critic(batch_s, batch_a)
            critic_loss = self.MseLoss(target_Q, current_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            # Compute the actor loss
            actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic networks
            for params in self.critic.parameters():
                params.requires_grad = True

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        for i, param_group in enumerate(self.actor_optimizer.param_groups):
            print(f'Param Group {i}, Learning Rate: {param_group["lr"]}')


