#!/bin/bash

# Script: sweep_stuff.sh

# Check if a Python command is provided
if [ $# -eq 0 ]; then
    echo "No Python command provided."
    exit 1
fi

# Hardcoded base command with all the provided flags
BASE_CMD="bsub -q short-gpu -gpu num=1:j_exclusive=yes:gmem=4G"
BASE_CMD+=" -R \"affinity[thread*4] select[mem>4096] rusage[mem=4096]\""
#BASE_CMD+=" -R \"select[hname!=dgxws01]\" -R \"select[hname!=dgxws02]\" -R \"select[hname!=agn04]\""
#BASE_CMD+=" -R \"select[hname!=agn05]\" -R \"select[hname!=hgn02]\" -R \"select[hname!=hgn03]\""
#BASE_CMD+=" -R \"select[hname!=hgn07]\" -R \"select[hname!=hgn08]\" -R \"select[hname!=hgn09]\""
#BASE_CMD+=" -R \"select[hname!=hgn10]\" -R \"select[hname!=hgn11]\" -R \"select[hname!=hgn12]\""
BASE_CMD+=" -o lsf/output/out.%J -e lsf/output/err.%J"

# The Python command part provided as an argument
PYTHON_CMD="$*"

module load miniconda/24.11_environmentally
conda activate eb_hyperacuity

# Loop to print the command with different seed values
for SEED in {42..46}
do
#for spike_num in {16,32,48,64,96,128,256}
#do
eval "$BASE_CMD $PYTHON_CMD --seed $SEED"
#done
done
