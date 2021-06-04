# ============================================================================
# Python script for SLURM submission
# ============================================================================
import copy
from datetime import datetime
from itertools import product
import os
import subprocess
import tempfile

# Experiment parent directory
Base_Dir = '/miniscratch/chenant/ant/sr_return/2021-05-04/08-35_sf_gradR_lr001'

# Experimental parameters to pass to hydra
Experiment_Params_Dict = {
    'hydra.run.dir': '$hydra_run_dir',
    'runner': 'batched_offline',
    'runner.n_steps': 5e6,
    'runner.kwargs.log_interval_episodes': 10,
    'runner.kwargs.store_checkpoint': True,
    'runner.kwargs.checkpoint_freq': 2000,
    'runner.kwargs.checkpoint_dir_path': './checkpoints/',
    'runner.kwargs.train_every_n_frames': 1,
    'runner.kwargs.train_iterations': 1,
    'runner.kwargs.batch_size': 32,
    'runner.kwargs.replay_start_buffer_size': 5000,
    'runner.buffer_kwargs.buffer_size': 100000,
    'env.kwargs.env_name': ['breakout', 'asterix', 'seaquest', 'freeway', 'space_invaders'],
    'algo': 'lsf_dqn',
    'algo.kwargs.discount_gamma': 0.99,
    'algo.kwargs.sf_lambda': [1.0],
    'algo.kwargs.start_epsilon': 1.0,
    'algo.kwargs.end_epsilon': 0.1,
    'algo.kwargs.initial_epsilon_length': 5000,
    'algo.kwargs.epsilon_anneal_length': 100000,
    'algo.kwargs.use_target_net': True,
    'algo.kwargs.policy_updates_per_target_update': 1000,
    'algo.kwargs.optim_kwargs.lr': [0.001],
    'algo.kwargs.sf_optim_kwargs.lr': [0.001],
    'algo.kwargs.reward_optim_kwargs.lr': [0.001],
    'algo.kwargs.optim_kwargs.alpha': 0.95,
    'algo.kwargs.sf_optim_kwargs.alpha': 0.95,
    'algo.kwargs.reward_optim_kwargs.alpha': 0.95,
    'model': 'lsf_q_network',
    'model.cls_string': 'LQNet_shareQR',
    'model.kwargs.sf_hidden_sizes': 'null',  # 'null' for None
    'model.kwargs.sf_grad_to_phi': False,
    '+model.kwargs.value_grad_to_phi': False,
    'model.kwargs.reward_grad_to_phi': True,
    'training.seed': [2, 5, 8],
}

# SLURM submission parameters
Slurm_Param = {
    '--cpus-per-task': 1,
    '--gres': 'gpu:1',
    '--partition': 'main,long',
    '--mem': '12G',
    '--time': '26:00:00',  # max '48:00:00'
    '--exclude': '',  # can be empty
}


# Helper method to make experiment sub-directory name
def get_rundir_name(d):
    """
    Helper method to construct the result sub-directory name based on the
    experiment parameters
    :param d: dictionary of experiment parameters
    :return: string of the sub-directory name
    """

    env_str = str(d['env.kwargs.env_name'])

    lr_fl = float(d['algo.kwargs.optim_kwargs.lr'])
    lr_str = str(lr_fl - int(lr_fl))[2:7]

    if d['algo'] == 'lsf_dqn':
        lamb_fl = float(d['algo.kwargs.sf_lambda'])
        if lamb_fl < 1.0:
            lamb_str = str(lamb_fl - int(lamb_fl))[2:4]
        else:
            lamb_str = '10'
    else:
        lamb_str = 'None'

    seed_int = int(d['training.seed'])
    seed_str = str(seed_int)

    hydra_time_str = '${now:%Y%m%d%H%M%S}'

    return f'{env_str}/lam{lamb_str}s{seed_str}_{hydra_time_str}'


# SLURM error and output parent directory
StdOut_Dir = "/miniscratch/chenant/ant/sr_return/stdout_error"

# The python training script to call
Py_Script = "/home/mila/c/chenant/repos/sr-return/nonlinear/train_nonlinear.py"

# Where to submit (['slurm', 'local']), only do local for testing
Submit_Location = 'slurm'


# ============================================================================
# Shouldn't have to touch the below (hopefully)
# ============================================================================

def params_combination_to_param_dict_list(multirun_param_dict):
    """
    Helper for turning a dictionary of list of params to sweep over to
    a list of individual dictionary of params
    :param multirun_param_dict:
    :return:
    """

    def dict_2_dict_list(d: dict) -> dict:
        """
        Convert all of a dict's 1-depth entries into lists
        E.g. 1 -> [1]
        """
        new_d = {}
        for k in d:
            cur_v = d[k]
            if isinstance(cur_v, list):
                new_d[k] = cur_v
            else:
                new_d[k] = [cur_v]
        return new_d

    def gen_combinations(d: dict):
        """
        Generator for one-deep combinations of lists of input dict
        """
        keys, values = d.keys(), d.values()
        values_choices = (v for v in values)
        for comb in product(*values_choices):
            yield dict(zip(keys, comb))

    mp_dict = dict_2_dict_list(multirun_param_dict)

    param_dict_list = []
    for c in gen_combinations(mp_dict):
        param_dict_list.append(c)

    return param_dict_list


def param_dict_to_py_command(param_dict):
    """
    Turn a param dict into a string python command to write to a bash script
     to be submitted
    """
    p_dict = copy.deepcopy(param_dict)

    # ==
    # Add any additional augmentation to the param dict ?

    # ==
    # Construct the python argument
    py_command = f"python -u {Py_Script} \\"
    for k in p_dict:
        py_command += f"\n\t{k}={p_dict[k]} \\"

    return py_command


def submit_slurm_job(exp_param_dict):
    # ==
    # Generate the inner script string
    script_lines = [
        '#!/bin/bash',
        '',
        '# (1.1) Load packages',
        'module load python/3.7',
        'module load python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0',
        '',
        '# (1.2) Load environment',
        'source /home/mila/c/chenant/venvs/rlpyt/bin/activate',
        '',
    ]

    run_dir_name = get_rundir_name(exp_param_dict)
    run_dir_path = os.path.join(Base_Dir, run_dir_name)
    script_lines.extend([
        f"hydra_run_dir='{run_dir_path}'",
        "",
    ])
    cur_py_command = param_dict_to_py_command(exp_param_dict)
    script_lines.append(cur_py_command)
    inner_script = '\n'.join(script_lines)

    # ==
    # Construct the outer submission arguments
    dir_day = Base_Dir.split('/')[-2].split('-')[-1]  # hard-coded date day
    dir_time = ''.join(
        Base_Dir.split('/')[-1].split('-')[0:2])  # hard-coded date time
    job_name = f'{dir_day}-{dir_time}_{exp_param_dict["env.kwargs.env_name"]}'

    cur_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    err_file = os.path.join(
        StdOut_Dir, f'{cur_datetime}_error_{job_name}.txt'
    )
    out_file = os.path.join(
        StdOut_Dir, f'{cur_datetime}_out_{job_name}.txt'
    )

    # Construct base slurm args
    slurm_args = ['sbatch']
    for k in Slurm_Param:
        slurm_args.append(k)
        slurm_args.append(Slurm_Param[k])
    slurm_args.extend(['--output', out_file])
    slurm_args.extend(['--error', err_file])
    slurm_args.extend(['--job-name', job_name])

    # Construct inner script and submit
    with tempfile.NamedTemporaryFile() as scriptfile:
        arg_script = bytearray(inner_script, "utf8")
        scriptfile.write(arg_script)
        scriptfile.flush()

        if Submit_Location == 'slurm':
            slurm_args.append(scriptfile.name)
            print(slurm_args)
            s_args = [str(e) for e in slurm_args]
        else:
            s_args = ['bash', scriptfile.name]

        subprocess.run(s_args)


if __name__ == "__main__":

    param_dict_list = params_combination_to_param_dict_list(Experiment_Params_Dict)

    for job_i, param_dict in enumerate(param_dict_list):
        print(f'=====\nSubmitting job [{(job_i + 1)}/{len(param_dict_list)}]')
        print(param_dict)
        submit_slurm_job(param_dict)
        print()
