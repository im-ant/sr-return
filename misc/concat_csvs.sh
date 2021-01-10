#!/bin/bash
# ============================================================================
# Vertically concatenate a list of .csv files.
# Assumes:
#   - All files have the same header at the first line
#   - All files have no trailing whitespace
# ============================================================================
set -e

# Get list of input arguments
IN_ARG="$@"

# ==
# Iterate over all files with counter
i=0
for filepath in $IN_ARG; do
  # ==
  # (For first file) Save header
  if [ $i == 0 ]; then
    awk 'NR == 1' $filepath
  fi

  # ==
  # (For all files) Save non header
  awk 'NR != 1' $filepath

  # Increment counter
  i=$((i+1));
done