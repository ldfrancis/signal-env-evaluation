import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import numpy as np

import os

from .dqn import DQN, Buffer
from rlsignal.env import Environment

import wandb

torch.random.manual_seed(0)
np.random.seed(0)


class DQNTrainer:
    def __init__(self, trainer_config) -> None:
        self.env = Environment(trainer_config["env_config"])
        self.eval_avg_travel_time = np.inf
        exp_config = trainer_config["exp_config"]
        dqn_config = trainer_config["dqn_config"]
        
        self.episodes = exp_config["episodes"]
        self.steps = exp_config['steps']
        self.eval_steps = exp_config['eval_steps']
        self.buffer_size = dqn_config['buffer_size']
        self.target_update = dqn_config["target_update"]
        self.batch_size = exp_config["batch_size"]
        self.epsilon = dqn_config["max_epsilon"]
        self.epsilon_decay = dqn_config["epsilon_decay"]
        self.min_epsilon = dqn_config["min_epsilon"]
        self.max_epsilon = dqn_config["max_epsilon"]
        self.gamma = dqn_config["gamma"]
        self.exp_name = exp_config["name"]
        self.learning_rate = exp_config["learning_rate"]
        self.learning_start = dqn_config["learning_start"]

        self.agents, self.buffers = [], []
        for i in range(self.env.n_intersections):
            cfg = {}
            cfg["in_features"] = self.env.ob_gens[i].ob_length
            cfg["num_actions"] = self.env.n_actions[i]
            self.agents += [DQN(cfg)]
            self.buffers += [Buffer(self.buffer_size, self.batch_size, (self.env.ob_gens[i].ob_length,))]

        self.optimizers = [torch.optim.Adam(agent.qnet.parameters(), lr=1e-4) for agent in self.agents]

         

    def __call__(self):
        self.wandb_log = {}
        self.num_updates = [0]*self.env.n_intersections

        for episode in range(1,self.episodes+1):
            self.env.metric.clear()
            obs = self.env.reset()
            qlosses = []
            while self.env.steps < self.steps:
                phases = self.env.get_phases()

                if self.env.steps < self.learning_start or self.epsilon > np.random.random():
                    actions = [agent.sample_action() for agent in self.agents]
                else:
                    actions = [agent.select_action(obs[i], phases[i]) for i,agent in enumerate(self.agents)]

                next_obs, rewards, dones, _ = self.env.step(actions)
                self.env.metric.update(rewards)

                next_phases = self.env.get_phases()
                for i,buffer in enumerate(self.buffers):
                    buffer.save(obs[i], phases[i], actions[i], rewards[i], next_obs[i], next_phases[i], dones[i])
                    # update model if enough samples are available
                    if buffer.can_sample:
                        samples = buffer.sample_batch()
                        obs = samples["obs"]
                        phases = samples["phases"]
                        actions = samples["actions"]
                        rews = samples["rewards"]
                        n_obs = samples["next_obs"]
                        n_phases = samples["next_phases"]
                        dones = samples["dones"]

                        cur_qvalues = self.agents[i].qnet(obs, phases).gather(1, actions.unsqueeze(1))
                        with torch.no_grad():
                            next_qvalues = self.agents[i].target_qnet(n_obs, n_phases).max(1, keepdim=True)[0].detach()
                            target = rews.unsqueeze(1) + self.gamma*next_qvalues*(1-dones).unsqueeze(1)
                            # out = self.agents[i].target_qnet(n_obs, n_phases)
                            # target = rews + self.gamma * torch.max(out, dim=1)[0]
                            # target_f = self.agents[i].qnet(obs, phases)
                            # for j, action in enumerate(actions):
                            #     target_f[j][action] = target[j]

                        loss = F.mse_loss(cur_qvalues, target, reduction="mean")
                        # loss = F.mse_loss(self.agents[i].qnet(obs, phases), target_f, reduction="mean")

                        if buffer.size > self.learning_start:
                            # loss = self.criterion(self.model(b_t, train=True), target_f)
                            self.optimizers[i].zero_grad()
                            loss.backward()
                            clip_grad_norm_(self.agents[i].qnet.parameters(), 5.0)
                            self.optimizers[i].step()

                        qlosses += [loss.item()]
                        self.num_updates[i] +=1

                        if self.num_updates[i] % self.target_update == 0:
                             self.agents[i].update_target_qnet()

                        self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

            obs = next_obs
            mean_qloss = np.mean(qlosses)
            self.wandb_log.update({
                "train_avg_travel_time": self.env.metric.real_average_travel_time(),
                "train_throughput": self.env.metric.throughput(),
                "train_rewards": self.env.metric.rewards(),
                "qloss": mean_qloss,
            })
            self.eval()
            if wandb.run: wandb.log(self.wandb_log)
            print(f"\nEpisode {episode}:", self.wandb_log)

            if self.wandb_log["eval_avg_travel_time"] < self.eval_avg_travel_time:
                self.eval_avg_travel_time = self.wandb_log["eval_avg_travel_time"] 
                for i, agent in enumerate(self.agents):
                    pth = f"models/{self.exp_name}_dqn"
                    if not os.path.exists(pth): os.makedirs(pth)
                    torch.save(agent.qnet.state_dict(), f"{pth}/{i}.pt")
    
    def eval(self):
        self.env.metric.clear()
        obs = self.env.reset()
        while self.env.steps < self.steps:
            phases = self.env.get_phases()
            actions = [agent.select_action(obs[i], phases[i]) for i,agent in enumerate(self.agents)]
            next_obs, rewards, dones, _ = self.env.step(actions)
            self.env.metric.update(rewards)
            obs = next_obs

        self.wandb_log.update(
             {
                "eval_avg_travel_time": self.env.metric.real_average_travel_time(),
                "eval_throughput": self.env.metric.throughput(),
                "eval_rewards": self.env.metric.rewards(),
            }
        )
        