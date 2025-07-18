#!/bin/bash

#SBATCH --job-name=ssd_log_generation      # Descriptive job name
#SBATCH --ntasks=1                         # One task (one Python process)
#SBATCH --cpus-per-task=10                 # XX CPU threads for that process
#SBATCH --mem=20G                          # Total memory for the job
#SBATCH --time=12:00:00                    # Wall time limit
#SBATCH --partition=cpu                    # Use the 'cpu' partition
#SBATCH --output=logs/%x_%j.out            # Save stdout to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err             # Save stderr to logs/jobname_jobid.err
#SBATCH --mail-type=BEGIN,END,FAIL         # Email on job state changes
#SBATCH --chdir=/work/alexkrau/projects/ssd_v2/AgentSimulator

# Load environment
source /work/alexkrau/miniconda3/etc/profile.d/conda.sh
conda activate agentsim

# Make sure logs directory exists
mkdir -p logs

# Run Python evaluation script with explicitly passed CPU count
python simulate.py --log_path raw_data/experiment_1_settings/experiment_1_bimp_log.csv --case_id case_id --activity_name activity --resource_name resource --end_timestamp end_time --start_timestamp start_time --num_simulations 10 --central_orchestration --num_cores=${SLURM_CPUS_PER_TASK}