import numpy as np
import torch
import torch.nn.functional as F

import random


class DQN:
    cfg = {
        "observation_space":None,
        "action_space":None,
        "buffer_size": None,
        "batch_size": None,
        "learning_rate": None,
        "learning_starts": None,
        "target_update_frequency": None,
        "start_epsilon": None,
        "min_epsilon": None,
        "episodes":None,
        "steps_per_episode": None,
        "exploration_fraction": None,
    }   
    def __init__(self, dqn_config) -> None:
        self.cfg = dqn_config
        self.reset()

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def act(self, obs):
        self.transition = []
        if self._mode == "train":
            if not hasattr(self, "_epsilon"): self._epsilon = self.cfg["start_epsilon"]
            if random.random() < self._epsilon:
                action = self.cfg["action_space"].sample()
                self.transition += [obs, action]
                return action
        
        with torch.no_grad():
            if isinstance(obs, dict):
                obs = {k:torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k,v in obs.items()}
            else:
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            a_values = self.qnet(obs)
            action = torch.argmax(a_values, dim=-1).item()
            self.transition += [obs, action]
            return action
        
    def observe(self, obs, rew, done):
        assert self._mode == "train"
        self.transition += [rew, obs, done]
        self.buffer.save(*self.transition)
        
    def sample_action(self):
        return self.cfg["action_space"].sample()
    
    def update(self, step=0):
        assert self._mode == "train"
        self._epsilon = self._epsilon_schedule(step)

        if not self.buffer.can_sample(self.cfg["batch_size"]):
            return
        #     batch_size = 1
        # else:
        #     batch_size = self.cfg["batch_size"]

        batch = self.buffer.sample_batch(self.cfg["batch_size"])

        with torch.no_grad():
            target_max, _ = self.target_qnet(batch["next_obs"]).max(dim=1)
            td_target = batch["rewards"].flatten() + self.cfg["gamma"]*target_max*(1-batch["dones"].flatten())
        
        old_val = self.qnet(batch["obs"]).gather(1, batch["actions"]).squeeze(1)
        loss = F.mse_loss(td_target, old_val)
        self.qlosses += [loss.item()]; self.qlosses = self.qlosses[-self.cfg["loss_agg_window"]:]
        self.stats["qloss"] = np.mean(self.qlosses)
        self.stats["epsilon"] = self._epsilon
        
        if step > self.cfg["learning_starts"]:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.cfg["target_update_frequency"] == 0:
                self.update_target_qnet()
            

    def _epsilon_schedule(self, step):
        start_eps = self.cfg["start_epsilon"]
        min_eps = self.cfg["min_epsilon"]
        duration = self.cfg["episodes"]*(self.cfg["steps_per_episode"]/self.cfg["delta_time"])*self.cfg["exploration_fraction"]
        slope = (min_eps - start_eps) / duration
        return max(slope*step + start_eps, self.cfg["min_epsilon"])
    
    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path)

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path))

    def update_target_qnet(self):
        for t_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            t_param.data.copy_(
                self.cfg["tau"]*param.data + (1.0-self.cfg["tau"])*t_param.data
            )

    def reset(self):
        self.qnet = QNetwork(self.cfg["observation_space"],self.cfg["action_space"])
        self.target_qnet = QNetwork(self.cfg["observation_space"],self.cfg["action_space"])
        self.update_target_qnet()
        self.target_qnet.eval()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.cfg["learning_rate"])

        self.buffer = Buffer(
            self.cfg["buffer_size"], 
            self.cfg["observation_space"],
        )

        self.stats = {
            "qloss":None
        }
        self.qlosses = []
        
        

# class Preprocessor(torch.nn.Module):
#     def __init__(self, input_shape, num_phases) -> None:
#         super().__init__()
#         self.num_phases = num_phases
    
#     def forward(self, obs, phases):
#         feature = torch.cat([obs, F.one_hot(phases, self.num_phases)], dim=-1)
#         return feature


class QNetwork(torch.nn.Module):
    def __init__(self, obs_space, act_space) -> None:
        super().__init__()
        self._obs_space = obs_space
        self._act_space = act_space
        if not isinstance(obs_space, dict):
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(obs_space.shape[0], 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, act_space.n)
            )
        else:
            self.layers1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(3, 16, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(np.prod(obs_space["obs"].shape), 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 20),
                torch.nn.ReLU(),
            )
            self.layers2 = torch.nn.Sequential(
                torch.nn.Linear(20+act_space.n, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, act_space.n)
            )

    def forward(self, obs):
        if isinstance(obs, dict):
            x1 = self.layers1(obs["obs"])
            x1 = torch.cat([x1, obs["phase"]], axis=1)
            values = self.layers2(x1)
        else:
            values = self.layers(obs)

        return values
            


class Buffer:
    def __init__(self, max_size, obs_space) -> None:
        if isinstance(obs_space, dict):
            self.obs = np.zeros((max_size, *obs_space["obs"].shape))
            self.next_obs = np.zeros((max_size, *obs_space["obs"].shape))
            self.phases = np.ones((max_size, *obs_space["phase"].shape), dtype=int)
            self.next_phases = np.ones((max_size, *obs_space["phase"].shape), dtype=int)
        else:
            self.obs = np.zeros((max_size, *obs_space.shape))
            self.next_obs = np.zeros((max_size, *obs_space.shape))

        self._obs_space = obs_space
        self.actions = np.zeros((max_size,1))
        self.rewards = np.zeros((max_size,1))
        self.dones = np.zeros((max_size,1))
        self.max_size = max_size
        self.idx, self.size = 0, 0
    
    def _incr_idx(self):
        self.idx = (self.idx+1)%self.max_size
        self.size = min(self.size + 1, self.max_size)

    def save(self, obs, action, reward, next_obs, done):
        if isinstance(obs, dict):
            self.obs[self.idx] = obs["obs"]
            self.phases[self.idx] = obs["phase"]
            self.next_obs[self.idx] = next_obs["obs"]
            self.next_phases[self.idx] = next_obs["phase"]
        else:
            self.obs[self.idx] = obs
            self.next_obs[self.idx] = next_obs

        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self._incr_idx()

    def sample_batch(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)

        obs = torch.tensor(self.obs[idxs], dtype=torch.float32)
        next_obs = torch.tensor(self.next_obs[idxs], dtype=torch.float32)
        if isinstance(self._obs_space, dict):
            obs = {
                "obs":obs,
                "phase":torch.tensor(self.phases[idxs], dtype=torch.float32)
            }
            next_obs = {
                "obs":next_obs,
                "phase":torch.tensor(self.next_phases[idxs], dtype=torch.float32),   
            }

        return dict(
            obs = obs,
            actions = torch.tensor(self.actions[idxs], dtype=torch.long),
            rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32),
            next_obs = next_obs,
            dones = torch.tensor(self.dones[idxs], dtype=torch.float32),
        )

    def can_sample(self, batch_size):
        return self.size > batch_size
    

def get_agent_class(name):
    if name == "dqn":
        return DQN