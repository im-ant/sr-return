# ============================================================================
# Training script with the rlpyt package
# ============================================================================

import sys
import random

import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from gym_minigrid import wrappers
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.atari import (AtariMixin, AtariFfAgent)
from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector,
                                                    CpuWaitResetCollector)
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils import seed as rlpyt_seed

from agents.categorical_lsf import CategoricalPgLsfAgent
from algos.a2c_lsf import A2C_LSF
from models.cat_lsf_ff_model import CategoricalPgLsfFfModel


def minigrid_env_f(name='MiniGrid-Empty-5x5-v0'):
    minigrid_env = gym.make(name)
    minigrid_env = wrappers.ImgObsWrapper(minigrid_env)

    return GymEnvWrapper(minigrid_env)


def build_and_train(cfg: DictConfig, cuda_idx=None, seed=None):
    # =====
    # Set up hardware
    n_parallel = 2
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")

    # =====
    # Set up environment sampler

    sampler = SerialSampler(
        EnvCls=minigrid_env_f,
        TrajInfoCls=TrajInfo,  # collect default trajectory info
        CollectorCls=CpuResetCollector,  # [CpuWaitResetCollector, CpuResetCollector]
        env_kwargs=cfg.environment.kwargs,
        eval_CollectorCls=SerialEvalCollector,
        eval_env_kwargs={},  # eval stuff, don't think it is used
        **cfg.sampler.kwargs,
    )

    # =====
    # Set up model and agent  # NOTE: no need to initialize env sizes?
    model_cls = globals()[cfg.model.cls_string]
    model_kwargs = cfg.model.kwargs

    agent_cls = globals()[cfg.agent.cls_string]
    agent = agent_cls(ModelCls=model_cls,
                      model_kwargs=model_kwargs,
                      initial_model_state_dict=None)

    # =====
    # Set up RL algorithm
    algo_cls = globals()[cfg.algo.cls_string]
    optim_cls = torch.optim.Adam  # maybe todo: make into config
    algo = algo_cls(OptimCls=optim_cls, **cfg.algo.kwargs)

    # =====
    # Set up runner
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        seed=seed,
        **cfg.runner.kwargs,
    )

    # ==
    log_params = dict(cfg.environment.kwargs)  # TODO not sure why useful
    run_ID = 0  # TODO remove? or change to seed
    name = "a2c_" + cfg.environment.kwargs.name  # TODO change
    with logger_context(run_ID=run_ID,
                        name=name,
                        log_params=log_params,
                        **cfg.logger_context.kwargs):
        runner.train()


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
        seed = rlpyt_seed.make_seed()
    rlpyt_seed.set_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # =====================================================
    # Run experiments
    build_and_train(cfg, cuda_idx=cuda_idx, seed=seed)



if __name__ == "__main__":
    main()
