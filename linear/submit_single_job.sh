#!/bin/bash
# ============================================================================
# Bash script to submit multiple jobs
set -e

# Optional: message to put into the created directory? TODO fix this
dir_message="Running first SF agent parameter sweep"

# The parent experimental directory, each experiment will be stored as a sub-
# directory within this directory
dir_path="/network/tmp1/chenant/ant/sr_trace/11-06/exp2_sf_runs"

# Path to the configuration file (same name as output dir)
config_file_path="$dir_path.ini"

# Experiment name (for the submission only)
job_name="exp2"

# How verbose to be in the stdout print line (per n episodes (0 for None), or "tqdm")
verbosity="50"

# Job file path
job_file="/home/mila/c/chenant/repos/sr-return/linear/arg-job_train-boyan.script"

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
time_per_job="12:00:00"




# ============================================================================
# Below is automatically ran

if [ -d "$dir_path" ]; then
  # Do not run job if same directory exists
  echo "Stopped. Directory already exists: $dir_path"
else
  # Make experimental directory
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
         --gres=$gres_per_job \
         --partition=$partition_per_job \
         --mem=$mem_per_job \
         --time=$time_per_job \
         --output="$cur_out_file" \
         --error="$cur_error_file" \
         --export=logpath="$dir_path",configpath="$config_file_path",verbosity="$verbosity" \
         --job-name=$job_name \
         $job_file

fi



