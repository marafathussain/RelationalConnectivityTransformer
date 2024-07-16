#!/bin/bash

#SBATCH --partition=fnndsc-gpu
#SBATCH --account=fnndsc
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Titan_RTX:1
#SBATCH --output=logs/rbnt4-%j.out

source /neuro/labs/grantlab/users/arafat.hussain/bnt/bin/activate
python -m source --multirun model=rbnt4 score=fiq
