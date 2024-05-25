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

from rsoccer_gym.ssl.ssl_go_to_ball.ssl_gym_go_to_ball import SSLGoToBallEnv

def evaluate(
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    actor_train=None,
    path='',
    global_step=0
):
    env = SSLGoToBallEnv(
        field_type=2
    )
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    model = Model(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    model.load_state_dict(actor_train.state_dict())
    model.eval()

    episodic_returns = []
    imgs = []
    obs, _ = env.reset()

    done = False
    reward = 0
    while not done:
        env.render()
        img = env.render(mode = 'rgb_array')
        imgs.append(img)

        actions = model(torch.Tensor(obs).to(device)).detach().cpu().numpy()

        next_obs, r, done, _, infos = env.step(actions)
        obs = next_obs
        reward += r
    episodic_returns.append(reward)

    print(f"\rAverage Reward: {np.vstack(episodic_returns).mean()}")

    # Salvar video
    output_filename = f'evaluate_{global_step}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_height, frame_width = imgs[0].shape[:2]
    out = cv2.VideoWriter(f'{path}/{output_filename}', fourcc, fps=30, frameSize=(frame_width, frame_height))

    for frame in imgs:
        out.write(frame[:,:, ::-1])

    out.release()

    return np.vstack(episodic_returns).mean()


@dataclass
class Args:
    exp_name: str = 'gotoball'#os.path.basename(__file__)[: -len(".py")]
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

def simulate(actor, args, idx):

    obs, _ = env.reset()
    w = 20
    steps_rewards = []
    for global_step in range(args.number_steps*idx, args.number_steps*(idx + 1)):

        if idx == 0 and global_step < args.learning_starts:
            actions = env.action_space.sample()
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device)).detach().cpu().numpy()
                actions += torch.normal(0, actor.action_scale * args.exploration_noise).detach().cpu().numpy()
                actions = actions.clip(-np.ones(act_size), np.ones(act_size))


        next_obs, rewards, terminations, _, infos = env.step(actions)
        steps_rewards.append(rewards)
        real_next_obs = next_obs.copy()

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        
        if terminations:
            writer.add_scalar("losses/rewards", np.sum(steps_rewards[0]), global_step)
            obs, _ = env.reset()
            steps_rewards = []
            terminations = False

        obs = next_obs
        print(f"\rStep: {global_step:<12}Train Rewards {np.mean(steps_rewards[-w:]):>5.2f}", end='')
    return rb

def learn(epoch, rb, qf1, qf1_target, actor, target_actor, args):

    for ep in range(epoch, epoch+5):
        ep_loss = {'q_value': 0, 'actor': 0}

        #n_batches = rb.buffer_size//args.batch_size
        for batch in range(args.n_batches):
            data = rb.sample(args.batch_size)
            qf1_loss = loss_q(data, qf1, qf1_target, target_actor)
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if (global_step+1) % args.policy_frequency == 0:
                actor_loss = loss_actor(data, qf1, actor)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            ep_loss['q_value'] += qf1_loss.item()
            ep_loss['actor'] += actor_loss.item()

        writer.add_scalar("losses/qf1_loss", qf1_loss.item()/args.n_batches, ep)
        writer.add_scalar("losses/actor_loss", actor_loss.item()/args.n_batches, ep)


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

    env = SSLGoToBallEnv(
        field_type=2
    )
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    rb = ReplayBuffer(
        args.buffer_size,
        Box(low=-env.field.length/2, high=env.field.length/2, shape=(obs_size, )),
        Box(low=-1, high=1, shape=(act_size, )),
        device,
        handle_timeout_termination=False,
    )

    actor = Actor(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    qf1 = QNetwork(act_size, obs_size).to(device)
    # a, q = torch.load('runs/ssl_pequi__ssl_ddpg__1__1716303010/ssl_ddpg.cleanrl_model')

    # actor.load_state_dict(a)
    # qf1.load_state_dict(q)

    qf1_target = QNetwork(act_size, obs_size).to(device)
    target_actor = Actor(act_size, obs_size, np.ones(act_size), -np.ones(act_size)).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    global_step = 0
    for idx in range(args.total_timesteps//args.number_steps):
        simulate(actor, args, idx)
        if global_step < args.learning_starts:
            learn(args.ep, rb, qf1, qf1_target, actor, target_actor, args)
        global_step += args.number_steps

        path=f"runs/{run_name}/video"

        if not os.path.exists(path):
            os.makedirs(path)

        if global_step%args.eval_gap == 0:
            episodic_return= evaluate(
                Model=Actor,
                device=device,
                actor_train=actor,
                path=path,
                global_step=global_step
            )

            writer.add_scalar(f"eval/episodic_reward", episodic_return, global_step)
            
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save((actor.state_dict(), qf1.state_dict()), model_path)

            print(f"model saved to {model_path}")
                
    env.close()
    writer.close()