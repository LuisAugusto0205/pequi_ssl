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
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

from rsoccer_gym.ssl.ssl_multi_agent.ssl_multi_agent import SSLMultiAgentEnv

from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)
import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os



# with open('log_multi_agent.txt', 'w') as f:
#     pass

class SSLMultiAgentEnvRllib(SSLMultiAgentEnv):
    def __init__(self, config):
        super().__init__(**config)


tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=4)
# parser.add_argument("--num-policies", type=int, default=2)
parser.add_argument("--num-cpus", type=int, default=4)

parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=100000000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=10000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=50000.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--checkpoint", type=str, default="", help="checkpoint path"
)
parser.add_argument(
    "--logdir", type=str, default="", help="checkpoint path"
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

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
            'field_type': 2
        })
        .framework("torch")
        .training(num_sgd_iter=10, use_gae=True)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
        .evaluation(evaluation_interval=5)
        )

    alg = config.build()
    if args.checkpoint != "":
        writer = SummaryWriter(log_dir=args.logdir)
        alg.restore(args.checkpoint)
    else:
        log_dir = "volume/log_tensor/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    with open('volume/step_checkpoint.txt', 'r') as file:
        last_epoch, _ = file.readline().split(';')
        last_epoch = int(last_epoch)
    
    with open('volume/best_step_checkpoint.txt', 'r') as file:
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
                        alg.save('volume/best_checkpoint/')
                        with open('volume/best_step_checkpoint.txt', 'w') as file:
                            file.write(f'{epoch};{best}')

                    with open('volume/step_checkpoint.txt', 'w') as file:
                        file.write(f"{epoch};{results['episode_reward_mean']}")

                if epoch % 100 == 0:  
                    alg.save('volume/last_checkpoint/')
                
            except Exception as e:
                print(e)
                with open('volume/step_checkpoint.txt', 'w') as file:
                    file.write(f'{epoch};{best}')
                continue
            
            pbar.update(1)


        

    print(results)
    ray.shutdown()