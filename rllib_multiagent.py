import argparse
import os
import yaml
import random
from collections import deque
import numpy as np
from typing import Dict

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)

from custom_torch_model import CustomFCNet
from action_dists import TorchBetaTest
from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

# RAY_PDB=1 python rllib_multiagent.py
# ray debug
def create_rllib_env(config):
    return SSLMultiAgentEnv(**config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if "blue" in agent_id:
        pol_id = "policy_blue"
    elif "yellow" in agent_id:
        pol_id = "policy_yellow"
    return pol_id

@ray.remote
class ScoreCounter:
    def __init__(self, maxlen):
        self.last100 = deque(maxlen=maxlen)
        self.last100.extend([0.0 for _ in range(maxlen)])
        self.maxlen = maxlen

    def append(self, s):
        self.last100.append(s)

    def reset(self):
        self.last100.extend([0.0 for _ in range(self.maxlen)])

    def get_score(self):
        return np.array(self.last100).mean()

class SelfPlayUpdateCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):

        super().__init__(legacy_callbacks_dict)

    def on_episode_start(
        self, *, worker, base_env, policies, episode: Episode, env_index: int, **kwargs
    ):

        episode.hist_data["score"] = []

    def on_episode_end(
        self, *, worker, base_env, policies, episode: Episode, **kwargs
    ) -> None:
        info_a = episode.last_info_for("blue_0")
        single_score = info_a["score"]["blue"] - info_a["score"]["yellow"]

        score_counter = ray.get_actor("score_counter")
        score_counter.append.remote(single_score)

    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when score is high enough
        """
        score_counter = ray.get_actor("score_counter")
        current_score = ray.get(score_counter.get_score.remote())

        info["result"]["custom_metrics"]["score"] = current_score

        if current_score > 0.6:
            print("---- Updating Opponent!!! ----")
            algorithm = info["algorithm"]
            algorithm.set_weights(
                {
                    "policy_yellow": algorithm.get_weights(["policy_blue"])["policy_blue"],
                }
            )
            score_counter = ray.get_actor("score_counter")
            score_counter.restart.remote()


if __name__ == "__main__":
    ray.init(num_cpus=6, num_gpus=1)

    with open("config.yaml") as f:
        # use safe_load instead load
        file_configs = yaml.safe_load(f)
    
    configs = {**file_configs["rllib"], **file_configs["PPO"]}

    counter = ScoreCounter.options(name="score_counter").remote(
        maxlen=file_configs["score_average_over"]
    )

    configs["env_config"] = file_configs["env"]

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env(configs["env_config"])
    obs_space = temp_env.observation_space["blue_0"]
    act_space = temp_env.action_space["blue_0"]
    temp_env.close()

    # Register the models to use.
    ModelCatalog.register_custom_action_dist("beta_dist", TorchBetaTest)
    ModelCatalog.register_custom_model("custom_vf_model", CustomFCNet)
    # Each policy can have a different configuration (including custom model).


    configs["callbacks"] = SelfPlayUpdateCallback
    configs["multiagent"] = {
        "policies": {
            "policy_blue": (None, obs_space, act_space, {}),
            "policy_yellow": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["policy_blue"],
    }
    configs["model"] = {
        "custom_model": "custom_vf_model",
        "custom_model_config": file_configs["custom_model"],
        "custom_action_dist": "beta_dist",
    }
    configs["env"] = "Soccer"

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config=configs,
        stop={
            # "timesteps_total": 16000000,
            # "time_total_s": 7200, #2h
            "time_total_s": 60*60, #2h
        },
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir="./",
        #resume=True,
        #restore="/home/luisaugusto/ray_results/PPO_selfplay_rec/PPO_Soccer_2903d_00000_0_2024-10-22_09-42-55/checkpoint_000000",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")



    # alg = config.build()
    # if args.checkpoint_task > -1:
    #     alg.restore(f"volume/last_checkpoint_gotoball_task{args.checkpoint_task}/" if not args.best else f"volume/best_checkpoint_gotoball_task{args.checkpoint_task}/")
    
    # writer = SummaryWriter(log_dir=f'volume/log_tensor/gotoball_task{args.task}/')

    # with open(f'volume/last_step_checkpoint_task{args.task}.txt', 'r') as file:
    #     last_epoch, _ = file.readline().split(';')
    #     last_epoch = int(last_epoch)
    
    # with open(f'volume/best_step_checkpoint_task{args.task}.txt', 'r') as file:
    #     _, best = file.readline().split(';')
    #     best = float(best)

    # with tqdm.tqdm(total=100000, initial=last_epoch) as pbar:

    #     for epoch in range(last_epoch, 100000):
    #         try:
    #             alg.train()
    #             if epoch % 10 == 0:
    #                 results = alg.evaluate()['evaluation']
    #                 writer.add_scalar('episode_reward_max', results['episode_reward_max'], epoch)
    #                 writer.add_scalar('episode_reward_min', results['episode_reward_min'], epoch)
    #                 writer.add_scalar('episode_reward_mean', results['episode_reward_mean'], epoch)
    #                 writer.add_scalar('episode_len_mean', results['episode_len_mean'], epoch)

    #                 if results['episode_reward_mean'] > best:  
    #                     best = results['episode_reward_mean']
    #                     alg.save(f'volume/best_checkpoint_gotoball_task{args.task}/')
    #                     with open(f'volume/best_step_checkpoint_task{args.task}.txt', 'w') as file:
    #                         file.write(f'{epoch};{best}')

    #                 with open(f'volume/last_step_checkpoint_task{args.task}.txt', 'w') as file:
    #                     file.write(f"{epoch};{results['episode_reward_mean']}")

    #             if epoch % 100 == 0:  
    #                 alg.save(f'volume/last_checkpoint_gotoball_task{args.task}/')
                
    #         except Exception as e:
    #             print(e)
    #             with open(f'volume/last_step_checkpoint_task{args.task}.txt', 'w') as file:
    #                 file.write(f'{epoch};{best}')
    #             continue
            
    #         pbar.update(1)


    # print(results)
    # ray.shutdown()
