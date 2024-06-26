# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

from gymnasium.wrappers import TransformObservation

#from rsoccer_gym.ssl.ssl_go_to_ball.ssl_gym_go_to_ball import SSLGoToBallEnv
from rsoccer_gym.ssl.ssl_go_to_ball.ssl_multi_agent import SSLMultiAgentEnv

def evaluate(
    eval_episodes: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    actor_train=None,
):
    # envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    env = SSLGoToBallEnv(n_robots_blue=3, n_robots_yellow=3, field_type=2)
    #env = TransformObservation(env, lambda obs: [((obs[0] - obs[4])**2 + (obs[1] - obs[5])**2)**0.5, obs[2], obs[3], obs[6], obs[7], obs[8], obs[9], obs[10]])
    obs, _ = env.reset()
    
    obs_size = np.array(env.observation_space.shape).prod()
    act_size = np.prod(env.action_space.shape)

    model = Model(act_size, obs_size, env.action_space.high, env.action_space.low).to(device)
    
    model.load_state_dict(actor_train.state_dict())
    model.eval()

    for ep in range(eval_episodes):
        episodic_returns = []
        obs, _ = env.reset()
        done = False
        while not done:
            env.render()
            reward = 0
            if random.random() < epsilon:
                actions = env.action_space.sample()
            else:
                actions = model(torch.Tensor(obs).to(device)).detach().cpu().numpy()
            next_obs, r, done, _, infos = env.step(actions)
            obs = next_obs
            reward += r
            episodic_returns.append(reward)

        print(f"\rAverage Reward {ep:0>2}: {np.array(episodic_returns).mean()}")

    return episodic_returns


@dataclass
class Args:
    exp_name: str = 'ssl_ddpg'#os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "ssl_pequi"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.001
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.01
    """the scale of exploration noise"""
    learning_starts: int = 2000
    """timestep to start learning"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


def make_env():#env_id, seed, idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
            
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.action_space.seed(seed)
        env = SSLGoToBallEnv(field_type=2)
        env = TransformObservation(env, lambda obs: [((obs[0] - obs[4])**2 + (obs[1] - obs[5])**2)**0.5, obs[2], obs[3], obs[6], obs[7], obs[8], obs[9], obs[10]])
        obs, _ = env.reset()
        env.observation_space = gym.spaces.Box(low=-env.NORM_BOUNDS, high=env.NORM_BOUNDS, shape=(len(obs), ), dtype=np.float32)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, act_size, obs_size):
        super().__init__()
        self.fc1 = nn.Linear(act_size + obs_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, act_size, obs_size, high, low):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc_mu = nn.Linear(4, act_size)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = SSLGoToBallEnv(field_type=2)
    #env = TransformObservation(env, lambda obs: [((obs[0] - obs[4])**2 + (obs[1] - obs[5])**2)**0.5, obs[2], obs[3], obs[6], obs[7], obs[8], obs[9], obs[10]])
    env = TransformObservation(env, lambda obs: [obs[0], obs[4], obs[1], obs[5]])
    obs, _ = env.reset()
    env.observation_space = gym.spaces.Box(low=-env.NORM_BOUNDS, high=env.NORM_BOUNDS, shape=(len(obs), ), dtype=np.float32)
    
    obs_size = np.array(env.observation_space.shape).prod()
    act_size = np.prod(env.action_space.shape)
    
    print(env.action_space.__class__)
    print(gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(act_size, obs_size, env.action_space.high, env.action_space.low).to(device)
    qf1 = QNetwork(act_size, obs_size).to(device)
    actor_params = torch.load('runs/ssl_pequi__ssl_ddpg__1__1712090842/ssl_ddpg.cleanrl_model')[0]
    actor.load_state_dict(actor_params)

    qf1_target = QNetwork(act_size, obs_size).to(device)
    target_actor = Actor(act_size, obs_size, env.action_space.high, env.action_space.low).to(device)

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    #env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    w = 20
    steps_rewards = []
    eps = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                actions = []
                action = actor(torch.Tensor(obs).to(device))
                action += torch.normal(0, actor.action_scale * args.exploration_noise)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                #print(actions.shape)

        next_obs, rewards, terminations, trunc, infos = env.step(action)
        steps_rewards.append(rewards)

        real_next_obs = next_obs.copy()

        rb.add(obs, real_next_obs, action, rewards, terminations, infos)

        if terminations:
            eps += 1
            writer.add_scalar("losses/rewards", np.sum(steps_rewards), eps)
            obs, _ = env.reset()
            steps_rewards = []
            terminations = False

        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if (global_step+1) % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                #print("SPS:", int(global_step / (time.time() - start_time)), end='')
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            
        print(f"\rStep: {global_step:<10}Train Rewards {np.mean(steps_rewards[-w:]):>5.2f}", end='')

        if global_step%50000 == 0:
            episodic_returns = evaluate(
                eval_episodes=3,
                Model=Actor,
                device=device,
                epsilon=0,
                actor_train=actor
            )
            for local_idx, episodic_return in enumerate(episodic_returns):
                global_idx = 3*(global_step//3000) + local_idx
                writer.add_scalar("eval/episodic_return", np.mean(episodic_return), global_idx)
                
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    torch.save((actor.state_dict(), qf1.state_dict()), model_path)
    print(f"model saved to {model_path}")
    env.close()
    writer.close()