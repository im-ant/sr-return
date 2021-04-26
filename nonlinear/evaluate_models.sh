#!/bin/bash

# ============================================================================
# Script for evaluating model
#
# Author: Anthony G. Chen
# ============================================================================


# ===========================
# Experimental set-up

# (1.1) Load packages
module load python/3.7
module load python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0

# (1.2) Load environment
# source $HOME/venvs/rlpyt/bin/activate
source /home/mila/c/chenant/venvs/rlpyt/bin/activate
base_py_dir="/home/mila/c/chenant/repos/sr-return/nonlinear"

# (2) no data to copy

# ==
# Set up to run

# Input dirs
in_dir_base='/miniscratch/chenant/ant/sr_return/'
in_dir_wc=$in_dir_base'2021-04-10/15-58-ldqn-lr4x/*/lam*'
model_append_wc='checkpoints/cpkt*steps-5000000*.pt'
cjson_append_wc='checkpoints/cpkt*.json'
config_append_wc='.hydra/config.yaml'

# Output dirs
base_run_dir='/miniscratch/chenant/ant/sr_return/2021-04-26/'
sub_dir='eval_${now:%H%M%S}'
run_parent_dir=$base_run_dir$sub_dir

python -u "$base_py_dir"/evaluate_models.py \
    hydra.run.dir=$run_parent_dir \
    path.dir_wc=$in_dir_wc \
    path.model_append_wc=$model_append_wc \
    path.config_append_wc=$config_append_wc \


# (4) Copy things over to scratch?
# cp $EXP_LOG_PATH /network/tmp1/chenant/tmp/
