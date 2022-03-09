#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH -e predict_structure_error.txt
#SBATCH -o predict_structure_out.txt

#SBATCH --qos=normal

#Limit the run to a single node
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=60000
module purge
module load AlphaFold/2.1.1-fosscuda-2020b
# module load AlphaFold/2.0.0-fosscuda-2020b
# module load AlphaFold/2.0.1-fosscuda-2020b
script_dir=/g/kosinski/geoffrey/af2_lasv_interactome/af2_lasv_L_apms/scripts/colabfold_trick/colabfold_trick

precomputed_dir=/scratch/gyu/af2_lasv_L_apms_precomputed
processed_dir=/scratch/gyu/af2_lasv_L_apms_result
output_dir=/scratch/gyu/af2_lasv_L_apms_result_colab_trick

python $script_dir/colab_trick.py --precomputed_dir=$precomputed_dir --processed_dir=$processed_dir --protein_names=O00571-fragment_1 --output_dir=$output_dir --num_cycle=6 --data_dir=/scratch/AlphaFold_DBs/
