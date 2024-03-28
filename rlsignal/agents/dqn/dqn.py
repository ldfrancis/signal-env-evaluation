import numpy as np
import torch
import torch.nn.functional as F


class DQN:
    def __init__(self, dqn_config) -> None:
        in_features = dqn_config["in_features"]
        num_actions = dqn_config["num_actions"]
        self.qnet = QNetwork(in_features, num_actions)
        self.target_qnet = QNetwork(in_features, num_actions)
        self.update_target_qnet()
        self.target_qnet.eval()

        self.num_actions = num_actions

    def select_action(self, obs, phase):
        with torch.no_grad():
            # phase_onehot = np.eye(self.num_actions)[phase]
            # feature = np.concatenate([obs, phase_onehot], axis=-1)
            # if len(feature.shape) == 1:
            #     feature = np.expand_dims(feature, 0)

            # feature = torch.FloatTensor(feature)
            obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32)
            phase = torch.tensor([phase], dtype=torch.long)
            a_values = self.qnet(obs, phase)

            action = torch.argmax(a_values, dim=-1).item()

            return action
        
    def sample_action(self):
        return np.random.choice(self.num_actions)
    
    def update_target_qnet(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        

class Preprocessor(torch.nn.Module):
    def __init__(self, input_shape, num_phases) -> None:
        super().__init__()
        self.num_phases = num_phases
    
    def forward(self, obs, phases):
        feature = torch.cat([obs, F.one_hot(phases, self.num_phases)], dim=-1)
        return feature


class QNetwork(torch.nn.Module):
    def __init__(self, in_features, num_actions) -> None:
        super().__init__()
        self.preprocessor = Preprocessor(in_features, num_actions)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features+num_actions, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_actions)
        )

    def forward(self, obs, phases):
        return self.layers(self.preprocessor(obs, phases))


class Buffer:
    def __init__(self, max_size, batch_size, obs_shape) -> None:
        self.obs = np.zeros((max_size, *obs_shape))
        self.phases = np.ones((max_size,), dtype=int)*-1
        self.next_obs =  np.zeros((max_size, *obs_shape))
        self.next_phases = np.ones((max_size,), dtype=int)*-1
        self.actions = np.zeros((max_size,))
        self.rewards = np.zeros((max_size,))
        self.dones = np.zeros((max_size,))
        self.max_size = max_size
        self.batch_size = batch_size
        self.idx = 0
        
    @property
    def size(self):
        return np.sum(self.phases != -1)
    
    def _incr_idx(self):
        self.idx = (self.idx+1)%self.max_size

    def save(self, obs, phase, action, reward, next_obs, next_phase, done):
        self.obs[self.idx] = obs
        self.phases[self.idx] = phase
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.next_phases[self.idx] = next_phase
        self.dones[self.idx] = done
        self._incr_idx()

    def sample_batch(self):
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        return dict(
            obs = torch.tensor(self.obs[idxs], dtype=torch.float32),
            phases = torch.tensor(self.phases[idxs], dtype=torch.long),
            actions = torch.tensor(self.actions[idxs], dtype=torch.long),
            rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32),
            next_obs = torch.tensor(self.next_obs[idxs], dtype=torch.float32),
            next_phases = torch.tensor(self.next_phases[idxs], dtype=torch.long),
            dones = torch.tensor(self.dones[idxs], dtype=torch.float32),
        )

    @property
    def can_sample(self):
        return self.size > self.batch_size


