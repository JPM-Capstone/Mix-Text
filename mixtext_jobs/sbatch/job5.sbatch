#!/bin/bash

#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --gres=gpu:1            
#SBATCH --partition=a100_1,a100_2,rtx8000
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=20:00:00
#SBATCH --job-name=Job5
#SBATCH --output=/home/as14229/NYU_HPC/MixText/jobs/logs/batch_1/job5.log

source ~/.bashrc
conda activate mixtext
cd /home/as14229/NYU_HPC/MixText/

python train.py labeled_100_idr_100