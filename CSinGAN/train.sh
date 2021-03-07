#!/bin/bash
#SBATCH -J lewis
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:1

module add gcc5/5.5.0 
module add python37 
module add cuda10.2/toolkit/10.2.89
#module add ml-pythondeps-py37-cuda10.2-gcc/4.0.8
module add cudnn7.6-cuda10.2/7.6.5.32
module add hdf5/1.10.1
module add nccl2-cuda10.2-gcc/2.6.4
#pip install opencv-python
#pip install torch==1.4.0
#pip install torchvision==0.5.0
python main_train.py