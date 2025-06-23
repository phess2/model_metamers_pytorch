#!/bin/bash
#SBATCH --job-name=met_resnet50
#SBATCH --output=outLogs/standard%A_%a.out
#SBATCH --error=outLogs/standard%A_%a.err
#SBATCH --mem=10G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --partition=mit_normal_gpu

module load miniforge

source activate /orcd/data/jhm/001/om2/rphess/projects/github.com/model_metamers_pytorch/conda_envs/modmetam

# Add the project root to Python path so imports work correctly
export PYTHONPATH="${PYTHONPATH}:/orcd/data/jhm/001/om2/rphess/projects/github.com/model_metamers_pytorch"

python model_analysis_folders/visual_networks/resnet50/make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py $SLURM_ARRAY_TASK_ID -I 3000 -N 8 -O sgd
