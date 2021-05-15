# ============================================================================
# Python script for SLURM submission
# ============================================================================
import copy
from datetime import datetime
from itertools import product
import os
import subprocess
import time
import tempfile

# Experiment parent directory
Base_Dir = '/scratch/achen43/sr-return/2021-05-15/16-55_dqn'

# Experimental parameters to pass to hydra
Experiment_Params_Dict = {
    'hydra.run.dir': '$hydra_run_dir',
    'env.game': ['BreakoutNoFrameskip-v4', 'FreewayNoFrameskip-v4'],
    'algo.agent_cls_string': 'DQNAgent',
    'optim.kwargs.lr': [6.25e-5],
    'training.seed': [2, 5, 8],
}

# SLURM submission parameters
Sbatch_Header_List = [
    '#SBATCH --account=rrg-bengioy-ad',  # Yoshua pays for job
    '#SBATCH --cpus-per-task=1',
    '#SBATCH --gres=gpu:1',
    '#SBATCH --mem=12G',
    '#SBATCH --time=23:59:00',  # specify job hours
]


# Helper method to make experiment sub-directory name
def get_rundir_name(d):
    """
    Helper method to construct the result sub-directory name based on the
    experiment parameters
    :param d: dictionary of experiment parameters
    :return: string of the sub-directory name
    """

    env_str = str(d['env.game'])
    ag_str = str(d['algo.agent_cls_string'])

    genlr_fl = float(d['optim.kwargs.lr'])
    genlr_str = str(genlr_fl - int(genlr_fl))[2:7]

    #if d['algo'] == 'lsf_dqn':
    #    lamb_fl = float(d['algo.kwargs.sf_lambda'])
    #    if lamb_fl < 1.0:
    #        lamb_str = str(lamb_fl - int(lamb_fl))[2:4]
    #    else:
    #        lamb_str = '10'
    #else:  # TODO: add this later?
    #   lamb_str = 'None'

    seed_int = int(d['training.seed'])
    seed_str = str(seed_int)

    hydra_time_str = '${now:%Y%m%d%H%M%S}'

    subdir_name = f'{env_str}/{ag_str}_s{seed_str}_{hydra_time_str}'

    return subdir_name


# SLURM error and output parent directory
StdOut_Dir = "/scratch/achen43/sr-return/slurm_out"

# The python training script to call
Py_Script = "/home/achen43/repos/sr-return/ShangtongZhang_DeepRL/train_q_network.py"

# Where to submit (['slurm', 'local']), only do local for testing
Submit_Location = 'slurm'

# Path to store the generated sbatch files during job submission (can be tmp)
SBatch_File_Dir = "/scratch/achen43/sr-return/gen_sbatch_files"


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

    # Header for job specifications
    script_lines = ['#!/bin/bash']
    script_lines.extend(Sbatch_Header_List)
    # Create environment
    script_lines.extend([
        '',
        '# (1.) Create environment',
        'module load python/3.8',
        'source /home/achen43/venvs/drl/bin/activate',
        '',
    ])

    # Generate data output directory
    run_dir_name = get_rundir_name(exp_param_dict)
    run_dir_path = os.path.join(Base_Dir, run_dir_name)
    script_lines.extend([
        f"hydra_run_dir='{run_dir_path}'",
        "",
    ])
    # Add the python command to run job
    cur_py_command = param_dict_to_py_command(exp_param_dict)
    script_lines.append(cur_py_command)
    # Construct the full 'inner' script to submit
    inner_script = '\n'.join(script_lines)

    # ==
    # Construct the outer submission arguments
    cur_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Hard-coded construction of job name
    dir_day = Base_Dir.split('/')[-2].split('-')[-1]  # hard-coded date day
    dir_time = ''.join(
        Base_Dir.split('/')[-1].split('-')[0:2])  # hard-coded date time
    job_name = f'{dir_day}-{dir_time}_{exp_param_dict["env.game"]}'

    # SLURM output and error files path (same file on CC)
    # NOTE:  %x -> job name, and %j -> job ID number
    out_file = os.path.join(
        StdOut_Dir, f'%x-%j_{cur_datetime}.out'
    )

    # Construct slurm submission args
    slurm_args = ['sbatch']
    slurm_args.extend(['--job-name', job_name])
    slurm_args.extend(['--output', out_file])

    # ==
    # Make inner script and submit
    scriptfile_stream = tempfile.NamedTemporaryFile(
        prefix=f'{job_name}_{cur_datetime}_',
        suffix='.script',
        dir=SBatch_File_Dir,
        delete=False,
    )

    arg_script = bytearray(inner_script, "utf8")
    scriptfile_stream.write(arg_script)
    scriptfile_stream.flush()  # Write inner script

    if Submit_Location == 'slurm':
        slurm_args.append(scriptfile_stream.name)
        print(slurm_args)
        s_args = [str(e) for e in slurm_args]
    else:
        s_args = ['bash', scriptfile_stream.name]
    subprocess.run(s_args)  # Submit inner script

    scriptfile_stream.close()


if __name__ == "__main__":

    param_dict_list = params_combination_to_param_dict_list(Experiment_Params_Dict)

    for job_i, param_dict in enumerate(param_dict_list):
        print(f'=====\nSubmitting job [{(job_i + 1)}/{len(param_dict_list)}]')
        print(param_dict)
        submit_slurm_job(param_dict)
        print()
        # In accordance with
        # https://docs.computecanada.ca/wiki/Running_jobs#Use_sbatch_to_submit_jobs
        time.sleep(1.0)
