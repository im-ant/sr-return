# ============================================================================
# Training script with the rlpyt package
# ============================================================================

import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

# from gym_minigrid import wrappers
from minatar import Environment  # MinAtar repo

from algos.ac_lambda import ACLambda
from models.ac_network import ACNetwork
from utils.runner import IncrementalOnlineRunner


def build_and_train(cfg: DictConfig, cuda_idx=None, seed=None):
    # ==
    # Set up device  TODO put here?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==
    # Set up environment
    env_cls = Environment
    env_kwargs = {
        'env_name': 'breakout',
        'random_seed': None,
    }

    # ==
    # Set up algorithm and model
    algo_cls = ACLambda
    algo_kwargs = {
        'ModelCls': ACNetwork,
        'model_kwargs': {},
    }
    proto_algo = algo_cls(**algo_kwargs)

    # ==
    # Start runner
    runner = IncrementalOnlineRunner(
        algo=proto_algo,
        EnvCls=env_cls,
        env_kwargs=env_kwargs,
        device=device,
    )

    runner.train(n_steps=1e6)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # =====================================================
    # Initialize GPU
    cuda_idx = 0 if torch.cuda.is_available() else None
    print('cuda_idx:', cuda_idx)

    # =====================================================
    # Set seeds
    seed = None  # TODO set seeds
    # TODO

    # =====================================================
    # Run experiments
    build_and_train(cfg, cuda_idx=cuda_idx, seed=seed)


if __name__ == "__main__":
    main()
