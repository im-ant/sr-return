# ============================================================================
# Training script with the rlpyt package
# ============================================================================


import sys

import gym
from gym_minigrid import wrappers
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.atari import (AtariMixin, AtariFfAgent)
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector,
                                                    CpuWaitResetCollector)
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


def minigrid_env_f(**kwargs):
    minigrid_env = gym.make('MiniGrid-Empty-5x5-v0')
    minigrid_env = wrappers.ImgObsWrapper(minigrid_env)

    return GymEnvWrapper(minigrid_env)


def build_and_train(cfg: DictConfig, cuda_idx=None):
    # =====
    # Set up hardware
    n_parallel=2
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")

    game = 'MiniGrid-Empty-5x5-v0'

    # =====
    # Set up environment sampler

    sampler = SerialSampler(
        EnvCls=minigrid_env_f,
        TrajInfoCls=TrajInfo,  # collect default trajectory info
        CollectorCls=CpuResetCollector,  # [CpuWaitResetCollector, CpuResetCollector]
        env_kwargs={},
        batch_T=10,  # seq length of per batch of sampled data
        batch_B=1,
        max_decorrelation_steps=0,
        eval_CollectorCls=SerialEvalCollector,
        eval_env_kwargs={},  # eval stuff, don't think it is used
        eval_n_envs=0,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )

    #example_env = minigrid_env_f()  # TODO delete this is not used

    # =====
    # Set up RL algorithm
    algo = A2C()  # Run with defaults.

    # =====
    # Set up model and agent  # NOTE: no need to initialize env sizes?
    model_kwargs = {
        'fc_sizes': 512,
        'use_maxpool': False,
        'channels': None,
        'kernel_sizes': [2, 1],
        'strides': [2, 1],
        'paddings': None,
    }
    agent = AtariFfAgent(model_kwargs=model_kwargs)


    # =====
    # Set up runner
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e4,
        affinity=affinity,
    )

    #tmp_k = model_kwargs  # TODO delete, these are just used to see the model size
    #tmp_k['image_shape'] = example_env.observation_space.shape
    #tmp_k['output_size'] = example_env.action_space.n
    #tmp = AtariFfModel(**tmp_k)
    #print(tmp)
    #a = 1/0

    # ==
    config = dict(game=game)
    run_ID = 0
    name = "a2c_" + game
    log_dir = "example_tmp"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))  # print the configs

    # =====================================================
    # Initialize GPU
    cuda_idx = 0 if torch.cuda.is_available() else None
    print('cuda_idx:', cuda_idx)

    # =====================================================
    # TODO set seeds

    # =====================================================
    # Run experiments
    build_and_train(cfg, cuda_idx=cuda_idx)
    pass



if __name__ == "__main__":
    main()
