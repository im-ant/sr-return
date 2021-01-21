# ============================================================================
# Training script for incremental online updates
# ============================================================================

import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import random
import torch

# from gym_minigrid import wrappers
from minatar import Environment  # MinAtar repo

from algos.ac_lambda import ACLambda
from algos.lsf_ac_lambda import LSF_ACLambda
from models.ac_network import ACNetwork
from models.lsf_ac_network import LSF_ACNetwork
from utils.runner import IncrementalOnlineRunner


def build_and_train(cfg: DictConfig, cuda_idx=None, seed=None):
    # ==
    # Set up environment
    env_cls = Environment
    env_kwargs = {
        'random_seed': seed,
        **cfg.env.kwargs,
    }

    # ==
    # Set up algorithm and model
    model_cls = globals()[cfg.model.cls_string]
    model_kwargs = {**cfg.model.kwargs}

    algo_cls = globals()[cfg.algo.cls_string]
    algo_kwargs = {
        'ModelCls': model_cls,
        'model_kwargs': model_kwargs,
        **cfg.algo.kwargs,
    }
    proto_algo = algo_cls(**algo_kwargs)

    # ==
    # Start runner
    runner = IncrementalOnlineRunner(
        algo=proto_algo,
        EnvCls=env_cls,
        env_kwargs=env_kwargs,
        device=cuda_idx,
        **cfg.runner.kwargs,
    )

    runner.train(n_steps=cfg.runner.n_steps)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))  # print the configs

    # =====================================================
    # Initialize GPU
    cuda_idx = 0 if torch.cuda.is_available() else None
    print('cuda_idx:', cuda_idx)

    # =====================================================
    # Set seeds
    seed = cfg.training.seed
    if seed is None:
        seed = 0
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # =====================================================
    # Run experiments
    build_and_train(cfg, cuda_idx=cuda_idx, seed=seed)


if __name__ == "__main__":
    main()
