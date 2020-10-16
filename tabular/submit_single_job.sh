#!/bin/bash
# ============================================================================
# Bash script to submit multiple jobs
set -e

# Optional: message to put into the created directory? TODO fix this
dir_message="Comparing SR and STrace agents, 5000 episodes discounted (gamma=0.8)"

# The parent experimental directory, each experiment will be stored as a sub-
# directory within this directory
dir_path="/network/tmp1/chenant/ant/exp_foward_trace/10-03/exp24_STrace_SR_comp_5000epis"

# Experiment name (for the submission only)
job_name="exp24"

# How verbose to be in the stdout print line (per n episodes (0 for None), or "tqdm")
verbosity="50"

# Job file path
job_file="/home/mila/c/chenant/repos/exp-foward-trace/tabular/arg-job_train-chain.script"

# Job partition (same for all jobs)
partition_per_job="main,long"

# Job resource (same for all jobs)
# pascal: [titanx, titanxp]; turing: [titanrtx, rtx8000];
# gres_per_job="gpu:pascal:1"

# Specify cpu need (same for all jobs)
cpu_per_task="1"

# Specify memory (RAM) need (same for all jobs)
mem_per_job="12G"

# Specify time need (same for all jobs)
time_per_job="12:00:00"




# ============================================================================
# Below is automatically ran

if [ -d "$dir_path" ]; then
  echo "Stopped. Directory already exists: $dir_path"
else
  #
  mkdir $dir_path
  echo "Made directory: $dir_path"

  # ==
  # Little print-out
  echo "Job submitted: `date`" > "$dir_path/submit_note.txt"
  echo $dir_message >> "$dir_path/submit_note.txt"

  # Create error and output files
  cur_error_file="$dir_path/error_$job_name.txt"
  cur_out_file="$dir_path/out_$job_name.txt"

  # ==
  # Submit job
  sbatch --cpus-per-task=$cpu_per_task \
         --partition=$partition_per_job \
         --mem=$mem_per_job \
         --time=$time_per_job \
         --output="$cur_out_file" \
         --error="$cur_error_file" \
         --export=logpath="$dir_path",verbosity="$verbosity" \
         --job-name=$job_name \
         $job_file

fi



