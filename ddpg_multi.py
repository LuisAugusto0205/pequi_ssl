# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.spaces import Box

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import time

import cv2

from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

def evaluate(
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    actor_train_blue=None,
    actor_train_yellow=None, 
    path='',
    global_step=0
):
    env = SSLMultiAgentEnv(
        n_robots_blue=1, 
        n_robots_yellow=0, 
        field_type=2, 
        max_ep_length=300
    )
    obs_size = env.obs_size
    act_size = env.act_size

    model_blue = Model(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    model_blue.load_state_dict(actor_train_blue.state_dict())
    model_blue.eval()

    model_yellow = Model(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    model_yellow.load_state_dict(actor_train_yellow.state_dict())
    model_yellow.eval()

    episodic_returns_blue = []
    episodic_returns_yellow = []
    imgs = []
    obs, _ = env.reset(idx=0, r=0.21, seed=1)

    done = False
    while not done:
        env.render()
        img = env.render(mode = 'rgb_array')
        imgs.append(img)

        reward_blue = np.zeros(env.n_robots_blue)
        reward_yellow = np.zeros(env.n_robots_yellow)
        actions = []

        for i in range(env.n_robots_blue):
            action = model_blue(torch.Tensor(obs[i]).to(device)).detach().cpu().numpy()
            actions.append(action)
        
        for i in range(env.n_robots_yellow):
            action = model_yellow(torch.Tensor(obs[env.n_robots_blue + i]).to(device)).detach().cpu().numpy()
            actions.append(action)
            
            actions = np.concatenate(actions)

        next_obs, r, done, _, infos = env.step(actions)
        obs = next_obs
        reward_blue += r[0]
        reward_yellow += r[1]
        episodic_returns_blue.append(reward_blue)
        episodic_returns_yellow.append(reward_yellow)

    print(f"\rAverage Reward\tblue: {np.vstack(episodic_returns_blue).mean()}\tyellow: {np.vstack(episodic_returns_yellow).mean()}")

    # Salvar video
    output_filename = f'evaluate_{global_step}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_height, frame_width = imgs[0].shape[:2]
    out = cv2.VideoWriter(f'{path}/{output_filename}', fourcc, fps=30, frameSize=(frame_width, frame_height))

    for frame in imgs:
        out.write(frame[:,:, ::-1])

    out.release()

    return episodic_returns_blue, episodic_returns_yellow


@dataclass
class Args:
    exp_name: str = '1_blue_next_ball'#os.path.basename(__file__)[: -len(".py")]
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
    total_timesteps: int = 10000000
    number_steps: int = 1
    n_batches: int = 1
    eval_gap: int =10000
    ep: int = 1
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.001
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 10000
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.01
    """the scale of exploration noise"""
    learning_starts: int = 100000
    """timestep to start learning"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, act_size, obs_size):
        super().__init__()
        self.fc1 = nn.Linear(act_size + obs_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, act_size, obs_size, high, low):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, act_size)
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


def loss_q(data, qf1, qf1_target, target_actor):
    with torch.no_grad():
        next_state_actions = target_actor(data.next_observations)
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

    qf1_a_values = qf1(data.observations, data.actions).view(-1)
    return F.mse_loss(qf1_a_values, next_q_value)

def loss_actor(data, qf1, actor):
    return -qf1(data.observations, actor(data.observations)).mean()

def simulate(actor_blue, actor_yellow, args, idx):

    obs, _ = env.reset(idx=0, r=0.21, seed=1)
    w = 20
    steps_rewards = []
    for global_step in range(args.number_steps*idx, args.number_steps*(idx + 1)):

        if idx == 0 and global_step < args.learning_starts:
            actions = env.action_space.sample()
        else:
            with torch.no_grad():
                actions_blue = actor_blue(torch.Tensor(obs[:env.n_robots_blue]).to(device)).detach().cpu().numpy()
                actions_blue += torch.normal(0, actor_blue.action_scale.repeat(env.n_robots_blue, 1) * args.exploration_noise).detach().cpu().numpy()
                actions_blue = actions_blue.clip(-np.ones((env.n_robots_blue, act_size)), np.ones((env.n_robots_blue, act_size)))
                
                actions_yellow = actor_yellow(torch.Tensor(obs[env.n_robots_blue:]).to(device)).detach().cpu().numpy()
                actions_yellow = actions_yellow.clip(-np.ones((env.n_robots_yellow, act_size)), np.ones((env.n_robots_yellow, act_size)))
                
                actions = np.concatenate([actions_blue, actions_yellow]).flatten()


        next_obs, rewards, terminations, trunc, infos = env.step(actions)
        steps_rewards.append(rewards)
        real_next_obs = next_obs.copy()

        for i in range(env.n_robots_blue):
            rb.add(obs[i], real_next_obs[i], actions[i*act_size:(i+1)*act_size], rewards[0][i], terminations, infos)
        
        if terminations:
            writer.add_scalar("losses/rewards", np.sum(steps_rewards[0]), global_step)
            obs, _ = env.reset(idx=0, r=0.21, seed=1)
            steps_rewards = []
            terminations = False

        obs = next_obs
        print(f"\rStep: {global_step:<12}Train Rewards {np.mean(steps_rewards[-w:]):>5.2f}", end='')
    return rb

def learn(epoch, rb, qf1_blue, qf1_blue_target, actor_blue, target_actor_blue, args):

    for ep in range(epoch, epoch+5):
        ep_loss = {'q_value': 0, 'actor': 0}

        #n_batches = rb_blue.buffer_size//args.batch_size
        for batch in range(args.n_batches):
            data_blue = rb.sample(args.batch_size)
            qf1_loss_blue = loss_q(data_blue, qf1_blue, qf1_blue_target, target_actor_blue)
            q_optimizer_blue.zero_grad()
            qf1_loss_blue.backward()
            q_optimizer_blue.step()

            if (global_step+1) % args.policy_frequency == 0:
                actor_loss_blue = loss_actor(data_blue, qf1_blue, actor_blue)
                actor_optimizer_blue.zero_grad()
                actor_loss_blue.backward()
                actor_optimizer_blue.step()

                for param, target_param in zip(actor_blue.parameters(), target_actor_blue.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1_blue.parameters(), qf1_blue_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            ep_loss['q_value'] += qf1_loss_blue.item()
            ep_loss['actor'] += actor_loss_blue.item()

        writer.add_scalar("losses/qf1_loss_blue", qf1_loss_blue.item()/args.n_batches, ep)
        writer.add_scalar("losses/actor_loss_blue", actor_loss_blue.item()/args.n_batches, ep)


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
    print(device)

    env = SSLMultiAgentEnv(n_robots_blue=1, n_robots_yellow=0, field_type=2, max_ep_length=300)
    obs_size = env.obs_size
    act_size = env.act_size

    rb = ReplayBuffer(
        args.buffer_size,
        Box(low=-env.field.length/2, high=env.field.length/2, shape=(obs_size, )),
        Box(low=-1, high=1, shape=(act_size, )),
        device,
        handle_timeout_termination=False,
    )

    actor_blue = Actor(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    qf1_blue = QNetwork(act_size, obs_size).to(device)
    # a, q = torch.load('runs/ssl_pequi__ssl_ddpg__1__1716303010/ssl_ddpg.cleanrl_model')

    # actor_blue.load_state_dict(a)
    # qf1_blue.load_state_dict(q)

    qf1_blue_target = QNetwork(act_size, obs_size).to(device)
    target_actor_blue = Actor(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    target_actor_blue.load_state_dict(actor_blue.state_dict())
    qf1_blue_target.load_state_dict(qf1_blue.state_dict())
    q_optimizer_blue = optim.Adam(list(qf1_blue.parameters()), lr=args.learning_rate)
    actor_optimizer_blue = optim.Adam(list(actor_blue.parameters()), lr=args.learning_rate)

    actor_yellow = Actor(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    actor_yellow.load_state_dict(actor_blue.state_dict())

    global_step = 0
    for idx in range(args.total_timesteps//args.number_steps):
        simulate(actor_blue, actor_yellow, args, idx)
        if global_step < args.learning_starts:
            learn(args.ep, rb, qf1_blue, qf1_blue_target, actor_blue, target_actor_blue, args)
        global_step += args.number_steps

        path=f"runs/{run_name}/video"

        if not os.path.exists(path):
            os.makedirs(path)

        if global_step%args.eval_gap == 0:
            episodic_returns_blue, episodic_returns_yellow = evaluate(
                eval_episodes=1,
                Model=Actor,
                device=device,
                epsilon=0,
                act_size=act_size,
                obs_size=obs_size,
                actor_train_blue=actor_blue,
                actor_train_yellow=actor_yellow,
                path=path,
                global_step=global_step
            )
            for idx, (episodic_return_blue, episodic_return_yellow) in enumerate(zip(episodic_returns_blue, episodic_returns_yellow)):

                writer.add_scalar(f"eval/episodic_return_blue_robot_{idx}", np.mean(episodic_return_blue), global_step)
                writer.add_scalar(f"eval/episodic_return_yellow_robot_{idx}", np.mean(episodic_return_yellow), global_step)
            
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save((actor_blue.state_dict(), qf1_blue.state_dict()), model_path)

            actor_yellow = Actor(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
            actor_yellow.load_state_dict(actor_blue.state_dict())
            print(f"model saved to {model_path}")
                
    env.close()
    writer.close()