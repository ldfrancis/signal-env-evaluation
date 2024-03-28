import torch
import numpy as np
from torch.distributions.categorical import Categorical


class PPO:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._observation_space = cfg["observation_space"]
        self._action_space = cfg["action_space"]
        self._act_lr = cfg["actor_learning_rate"]
        self._crit_lr = cfg["critic_learning_rate"]
        self._gamma = cfg["gamma"]
        self._gae_lambda = cfg["gae_lambda"]
        self._update_epochs = cfg["update_epochs"]
        self._norm_adv = cfg["norm_adv"]
        self._clip_coef = cfg["clip_coef"]
        self._clip_vloss = cfg["clip_vloss"]
        self._ent_coef = cfg["ent_coef"]
        self._vf_coef = cfg["vf_coef"]
        self._max_grad_norm = cfg["max_grad_norm"]
        self._target_kl = cfg["target_kl"]
        self._batch_size = cfg["batch_size"]
        self.reset()

    def reset(self):
        self.actor = Actor(self._observation_space, self._action_space)
        self.critic = Critic(self._observation_space)
        self._actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._act_lr)
        self._critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._crit_lr)
        self._buffer = Buffer()
        self.stats = {

        }
        self._lr_updated = False

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def _is_training(self):
        return self._mode == "train"

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs)
            probs = Categorical(logits=logits)
            action = probs.sample()
            if self._is_training:
                self._step_obs = obs
                self._step_action = action.item()
                self._step_logprob = probs.log_prob(action).item()
                self._step_entropy = probs.entropy().item()
                self._step_value = self.critic(obs).item()
        return action.item()

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)
    
    def get_value(self, obs):
        return self.critic(obs)

    def observe(self, obs, rew, done):
        self._buffer.save(
            self._step_obs, self._step_action, self._step_logprob, rew, self._step_value
            )
        
        if done:
            with torch.no_grad():
                next_value = self.get_value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            self._buffer.end_trajectory(obs, next_value, done, self._gamma, self._gae_lambda)

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))

    def update(self, step=0):
        if not self._lr_updated and step >= (self._cfg["episodes"]*self._cfg["steps_per_episode"]/self._cfg["delta_time"])*0.5:
            self._actor_optimizer.param_groups[0]['lr'] = 0.01
            self._critic_optimizer.param_groups[0]['lr'] = 0.01
            self._lr_updated = True
            # for g in self._actor_optimizer.param_groups:
            #     g['lr'] = 0.01
            # for g in self._critic_optimizer.param_groups:
            #     g["lr"] = g['lr']*0.01
        clipfracs = []
        if not self._buffer.trajectory_ended:
            return 
        for e in range(self._update_epochs):
            t_steps = self._buffer.size//self._batch_size + min(1,self._buffer.size%self._batch_size)
            for st in range(t_steps):
                obs, actions, logprobs, returns, advantages, values = self._buffer.sample(self._batch_size)

                _, newlogprob, entropy, newvalue = self.get_action_and_value(obs, actions)
                logratio = newlogprob - logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio-1) - logratio).mean()
                    clipfracs += [(((ratio-1)).abs() > self._clip_coef).float().mean().item()]

                if self._norm_adv:
                    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

                # Policy loss
                actor_loss = torch.max(
                    -advantages*ratio +
                    -advantages*torch.clamp(ratio, 1-self._clip_coef, 1+self._clip_coef)
                ).mean()

                # Value loss
                if self._clip_vloss:
                    v_loss_unclipped = (newvalue - returns)**2
                    v_clipped = values + torch.clamp(
                        newvalue - values,
                        -self._clip_coef,
                        self._clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns)**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5*v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue-returns)**2).mean()

                entropy_loss = entropy.mean()
                loss = actor_loss - self._ent_coef * entropy_loss + v_loss * self._vf_coef

                self._actor_optimizer.zero_grad()
                self._critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._max_grad_norm)
                self._actor_optimizer.step()
                self._critic_optimizer.step()

            if self._target_kl is not None and approx_kl > self._target_kl:
                break
        
        self.stats = {
            "critic/value_loss": v_loss.item(),
            "critic/learning_rate": self._critic_optimizer.param_groups[0]["lr"],
            "actor/learning_rate": self._actor_optimizer.param_groups[0]["lr"],
            "actor/policy_loss": actor_loss.item(),
            "actor/entropy": entropy_loss.item(),
            "actor/old_approx_kl": old_approx_kl.item(),
            "actor/approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),

        }







class Critic(torch.nn.Module):
    def __init__(self, observation_space) -> None:
        super().__init__()
        in_features = (
            np.prod(observation_space.shape) 
            if not isinstance(observation_space, dict) else 
            np.prod(observation_space["obs"].shape) + np.prod(observation_space["phase"])
        )
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, obs):
        in_features = (
            obs if not isinstance(obs, dict) else (
                torch.cat([obs["obs"], obs["phase"]], dim=1)
            )
        )
        return self.encode(in_features)
    


class Actor(torch.nn.Module):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        in_features = (
            np.prod(observation_space.shape) 
            if not isinstance(observation_space, dict) else 
            np.prod(observation_space["obs"].shape) + np.prod(observation_space["phase"])
        )
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space.n)
        )

    def forward(self, obs):
        in_features = (
            obs if not isinstance(obs, dict) else (
                torch.cat([obs["obs"], obs["phase"]], dim=1)
            )
        )
        return self.encode(in_features)
    

class Buffer:
    def __init__(self) -> None:
        self._obs = []
        self._actions = []
        self._logprobs = []
        self._rewards = []
        self._values = []
        self.trajectory = []

    @property
    def size(self):
        return len(self._obs)
    
    def save(self, obs, action, logprob, reward, value):
        self.trajectory += [(obs, action, logprob, reward, value)]
        self.trajectory_ended = False

    def end_trajectory(self, obs, value, done, gamma, gae_lambda):
        advantages = torch.zeros(len(self.trajectory))
        lastgaelam = 0
        obs, actions, logprobs, rewards, values = list(zip(*self.trajectory))

        obs = torch.cat(obs, dim=0)
        actions = torch.tensor(actions, dtype=torch.long)
        logprobs = torch.tensor(logprobs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        steps = len(obs)
        with torch.no_grad():
            for t in reversed(range(steps)):
                if t == steps - 1:
                    next_values = value
                else:
                    next_values = values[t+1]
                delta = rewards[t] + gamma * next_values - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam

            returns = advantages + values

        self._obs = obs
        self._actions = actions
        self._logprobs = logprobs
        self._rewards = rewards
        self._values = values
        self._returns = returns
        self._advantages = advantages
        
        self.trajectory = []
        self.trajectory_ended = True

    def sample(self, batch_size):
        bidxs = np.random.choice(self.size, self.size, replace=False)
        return (
            self._obs[bidxs], 
            self._actions[bidxs],
            self._logprobs[bidxs],
            self._returns[bidxs],
            self._advantages[bidxs],
            self._values[bidxs],
        )


