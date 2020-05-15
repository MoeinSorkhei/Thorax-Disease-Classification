#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem  10GB
#SBATCH --cpus-per-task 1
#SBATCH --constrain "shire|khazadum|belegost"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user sorkhei@kth.se
#SBATCH --time 24:00:00
#SBATCH --output /Midgard/home/%u/Thorax/logs/%J.out
#SBATCH --error  /Midgard/home/%u/Thorax/logs/%J.err


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
# shellcheck disable=SC1090
. ~/miniconda3/etc/profile.d/conda.sh
conda activate workshop

# shellcheck disable=SC2164
cd Thorax/src

python3 main.py \
      --model resnet \
      --lr 2e-6 \
      --freezed \
      --data_folder /local_storage/datasets/moein/Thorax/extracted/images \
      --checkpoints_path /Midgard/home/sorkhei/Thorax/checkpoints  \
      --use_comet