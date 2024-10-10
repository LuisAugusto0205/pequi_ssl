"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

import argparse
import os
import random

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)

from ray.tune.registry import (
    RLLIB_MODEL,
    RLLIB_ACTION_DIST,
    _global_registry,
)

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved

from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)
import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

class SSLMultiAgentEnvRllib(SSLMultiAgentEnv):
    def __init__(self, config):
        super().__init__(**config)

parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=8,  help="Número de cpus a serem usadas")

parser.add_argument(
    "--task", type=int, default=1, help="Número da task a ser treinada"
)

parser.add_argument(
    "--checkpoint_task", type=int, default=-1, help="Número da task a qual o checkpoint será usado para treinar a task atual"
)

parser.add_argument(
    "--best", action='store_true', help="Se presente usa o best checkpoint, caso contrário usa o last"
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus, num_gpus=1)

    # Register the models to use.
    ModelCatalog.register_custom_model("model_blue", TorchFullyConnectedNetwork)
    ModelCatalog.register_custom_model("model_yellow", TorchFullyConnectedNetwork)
    # Each policy can have a different configuration (including custom model).
    def gen_policy(team):

        config = PPOConfig.overrides(
            model={
                "custom_model": f"model_{team}",
                "fcnet_hiddens": [400, 300]
            },
            gamma=0.99
        )
        return PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(team): gen_policy(team) for team in ['blue', 'yellow']}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        if "blue" in agent_id:
            pol_id = "policy_blue"
        elif "'yellow" in agent_id:
            pol_id = "policy_yellow"
        return pol_id

    config = (
        PPOConfig()
        .environment(SSLMultiAgentEnvRllib, env_config={
            'n_robots_blue': 1, 
            'n_robots_yellow': 0,
            'field_type': 2,
            'random_pos_ball': True if args.task == 3 else False,
            'random_pos_robot': True if args.task in [2, 3] else False,
            'random_theta': True
        })
        .framework("torch")
        .training(num_sgd_iter=10, use_gae=True)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
        .evaluation(evaluation_interval=5)
        )

    alg = config.build()
    if args.checkpoint_task > -1:
        alg.restore(f"volume/last_checkpoint_gotoball_task{args.checkpoint_task}/" if not args.best else f"volume/best_checkpoint_gotoball_task{args.checkpoint_task}/")
    
    writer = SummaryWriter(log_dir=f'volume/log_tensor/gotoball_task{args.task}/')

    with open(f'volume/last_step_checkpoint_task{args.task}.txt', 'r') as file:
        last_epoch, _ = file.readline().split(';')
        last_epoch = int(last_epoch)
    
    with open(f'volume/best_step_checkpoint_task{args.task}.txt', 'r') as file:
        _, best = file.readline().split(';')
        best = float(best)

    with tqdm.tqdm(total=100000, initial=last_epoch) as pbar:

        for epoch in range(last_epoch, 100000):
            try:
                alg.train()
                if epoch % 10 == 0:
                    results = alg.evaluate()['evaluation']
                    writer.add_scalar('episode_reward_max', results['episode_reward_max'], epoch)
                    writer.add_scalar('episode_reward_min', results['episode_reward_min'], epoch)
                    writer.add_scalar('episode_reward_mean', results['episode_reward_mean'], epoch)
                    writer.add_scalar('episode_len_mean', results['episode_len_mean'], epoch)

                    if results['episode_reward_mean'] > best:  
                        best = results['episode_reward_mean']
                        alg.save(f'volume/best_checkpoint_gotoball_task{args.task}/')
                        with open(f'volume/best_step_checkpoint_task{args.task}.txt', 'w') as file:
                            file.write(f'{epoch};{best}')

                    with open(f'volume/last_step_checkpoint_task{args.task}.txt', 'w') as file:
                        file.write(f"{epoch};{results['episode_reward_mean']}")

                if epoch % 100 == 0:  
                    alg.save(f'volume/last_checkpoint_gotoball_task{args.task}/')
                
            except Exception as e:
                print(e)
                with open(f'volume/last_step_checkpoint_task{args.task}.txt', 'w') as file:
                    file.write(f'{epoch};{best}')
                continue
            
            pbar.update(1)


    print(results)
    ray.shutdown()
