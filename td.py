import torch.nn.functional as F
import numpy as np
import torch
import copy
import torch.nn as nn



# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
	
class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=device)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=device)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=device)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=device)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=device)

	def store(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(device)
		self.a[self.ptr] = a.to(device) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(device)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=device, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.sigmoid(self.l3(a))# * self.maxaction
        # a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Double_Q_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width) 
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
	


class TD3():
	def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-4, gamma=0.99, tau=0.005, K_epochs=50, net_width=256):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.max_action = max_action
		self.policy_noise = 0.2*self.max_action
		self.noise_clip = 0.5*self.max_action
		self.tau = tau
		self.delay_counter = 0
		self.gamma = gamma
		self.explore_noise=0.15
		self.action_dim = action_dim
		self.batch_size = 256
		self.K_epochs = K_epochs
		self.delay_freq = 1

		self.actor = Actor(state_dim, action_dim, net_width, self.max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Double_Q_Critic(state_dim, action_dim, net_width).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=lr_critic)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e4), dvc=device)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
				return (a + noise).clip(0, self.max_action)

	def update(self):
		for _ in range(self.K_epochs):
			self.delay_counter += 1
			with torch.no_grad():
				s, a, r, s_next, dw = self.buffer.sample(self.batch_size)

				# Compute the target Q
				target_a_noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
				'''↓↓↓ Target Policy Smoothing Regularization ↓↓↓'''
				smoothed_target_a = (self.actor_target(s_next) + target_a_noise).clamp(-self.max_action, self.max_action)
				target_Q1, target_Q2 = self.q_critic_target(s_next, smoothed_target_a)
				'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = r + (~dw) * self.gamma * target_Q  #dw: die or win

			# Get current Q estimates
			current_Q1, current_Q2 = self.q_critic(s, a)

			# Compute critic loss, and Optimize the q_critic
			q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			self.q_critic_optimizer.zero_grad()
			q_loss.backward()
			self.q_critic_optimizer.step()

			'''↓↓↓ Clipped Double Q-learning ↓↓↓'''
			if self.delay_counter > self.delay_freq:
				# Update the Actor
				a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
				self.actor_optimizer.zero_grad()
				a_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				with torch.no_grad():
					for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
						target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

					for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
						target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				self.delay_counter = 0

	def save(self, path):
		torch.save(self.actor.state_dict(), "{}/actor.pth".format(path))
		torch.save(self.q_critic.state_dict(), "{}/q_critic.pth".format(path))

	def load(self, path):
		self.actor.load_state_dict(torch.load("{}/actor.pth".format(path)))
		self.q_critic.load_state_dict(torch.load("{}/q_critic.pth".format(path)))
