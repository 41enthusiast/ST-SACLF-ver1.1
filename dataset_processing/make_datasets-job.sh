#!/bin/bash

#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu:ampere:1 # Request one gpu
 
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=short # Use the short QOS
#SBATCH -t 1-0 # Set maximum walltime to 1 day
#SBATCH --job-name=paintings-sweep# Name of the job
#SBATCH --mem=16G # Request 16Gb of memory
#SBATCH --exclude=gpu[0-8]

#SBATCH -o program_output.txt
#SBATCH -e whoopsies.txt

# Load the global bash profile
source /etc/profile
module load cuda/11.0

# Load your Python environment
source ../mv_test1/bin/activate

# Run the code

python gen_8_way_kaokore.py