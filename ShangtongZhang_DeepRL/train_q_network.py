#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import hydra
from omegaconf import DictConfig, OmegaConf

from torch.optim import RMSprop, Adam

from deep_rl import *


def train_dqn_pixel(cfg: DictConfig):
    config = Config()

    # Environment
    config.game = cfg.env.game
    config.task_fn = lambda: Task(cfg.env.game)
    config.eval_env = config.task_fn()

    # Experiment
    config.log_interval = cfg.experiment.log_interval
    config.save_interval = cfg.experiment.save_interval
    config.eval_interval = cfg.experiment.eval_interval

    # General training
    config.max_steps = int(cfg.training.max_steps)
    config.discount = cfg.training.discount
    config.history_length = cfg.training.history_length

    # Network
    config.network_fn = lambda: VanillaNet(
        config.action_dim,
        NatureConvBody(in_channels=config.history_length)
    )

    # Policy
    config.random_action_prob = LinearSchedule(
        **cfg.training.random_action_schedule
    )

    config.exploration_steps = cfg.training.exploration_steps
    config.async_actor = cfg.training.async_actor

    # Optimizer
    optim_cls = globals()[cfg.optim.cls_string]
    config.optimizer_fn = lambda params: optim_cls(
        params, **cfg.optim.kwargs
    )

    config.sgd_update_frequency = cfg.optim.sgd_update_frequency
    config.gradient_clip = cfg.optim.gradient_clip

    # Replay
    config.replay_cls = UniformReplay
    config.batch_size = cfg.replay_buffer.kwargs.batch_size
    config.n_step = cfg.replay_buffer.kwargs.n_step
    config.async_replay = cfg.replay_buffer.asynch

    replay_kwargs = dict(
        history_length=config.history_length,
        discount=config.discount,
        **cfg.replay_buffer.kwargs,
    )
    config.replay_fn = lambda: ReplayWrapper(
        replay_cls=config.replay_cls,
        replay_kwargs=replay_kwargs,
        asynch=config.async_replay
    )
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    # Other tricks
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = int(cfg.training.target_network_update_freq)
    config.double_q = cfg.training.double_q

    # Run
    agent_cls = globals()[cfg.algo.agent_cls_string]
    agent_obj = agent_cls(config)
    run_steps(agent_obj, cfg)


@hydra.main(config_path="conf_hydra", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))  # print the configs

    # Initialize GPU
    cuda_idx = 0 if torch.cuda.is_available() else None
    select_device(cuda_idx)  # -1 for CPU, positive in for GPU index
    print('cuda_idx:', cuda_idx)

    # Log files?
    mkdir('log')
    mkdir('tf_log')
    # Here because (https://github.com/ShangtongZhang/DeepRL/issues/63)
    set_one_thread()

    # Seed
    random_seed(seed=cfg.training.seed)

    # Run training
    train_dqn_pixel(cfg)


if __name__ == '__main__':
    main()
