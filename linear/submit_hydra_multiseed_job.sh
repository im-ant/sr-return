#!/bin/bash
# ============================================================================
# Bash script to submit multiple jobs via hydra.
# Should not have to change this
# ============================================================================
set -e


# Experiment name (for the submission only)
job_name="linear_hydra"

# Error file output parent directory
stdout_dir="/network/tmp1/chenant/ant/sr_trace/stdout_error"

# Job file path
job_file="/home/mila/c/chenant/repos/sr-return/linear/arg-job_train_hydra.script"

# Job partition (same for all jobs)
partition_per_job="main,long"

# Job resource (same for all jobs)
# pascal: [titanx, titanxp]; turing: [titanrtx, rtx8000];
# gres_per_job="gpu:pascal:1"
gres_per_job="gpu:1"

# Specify cpu need (same for all jobs)
cpu_per_task="1"

# Specify memory (RAM) need (same for all jobs)
mem_per_job="12G"

# Specify time need (same for all jobs)
time_per_job="36:00:00"

# Specify the list of seeds
seeds_list=(
'2' '4' '6' '8' '10' '12' '14' '16' '18' '20'
)



# ============================================================================
# Below is automatically ran

# Make the error and output files
cur_datetime=$(date '+%Y%m%d_%H%M%S')
cur_error_file="$stdout_dir/${cur_datetime}_error_$job_name.txt"
cur_out_file="$stdout_dir/${cur_datetime}_out_$job_name.txt"

for s in "${seeds_list[@]}"; do
  cur_seeds="$s"
  sbatch --cpus-per-task=$cpu_per_task \
         --gres=$gres_per_job \
         --partition=$partition_per_job \
         --mem=$mem_per_job \
         --time=$time_per_job \
         --output="$cur_out_file" \
         --error="$cur_error_file" \
         --export=seeds="$cur_seeds" \
         --job-name=$job_name \
         $job_file
done



#sbatch --cpus-per-task=$cpu_per_task \
#       --gres=$gres_per_job \
#       --partition=$partition_per_job \
#       --mem=$mem_per_job \
#       --time=$time_per_job \
#       --output="$cur_out_file" \
#       --error="$cur_error_file" \
#       --job-name=$job_name \
#       $job_file




