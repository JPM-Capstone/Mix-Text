#!/bin/bash

#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --gres=gpu:1            
#SBATCH --partition=a100_1,a100_2,rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=16:00:00
#SBATCH --job-name=Job15
#SBATCH --output=/home/as14229/NYU_HPC/Mix-Text/mixtext_jobs/logs/batch_1/job15.log

source ~/.bashrc
conda activate uda
cd /home/as14229/NYU_HPC/Mix-Text/

python train.py labeled_1000_idr_100