#!/bin/sh
#SBATCH --job-name=20250715_HaCaT_all_datasets_cell_paint
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3078

module load Singularity/3.6.4
module load Java/21.0.2

SEEDFILE=/nemo/project/proj-prosperity/hts/raw/projects/20250715_HaCaT_all_datasets_cell_paint/cellprofiler/cp_commands.txt
SEED=$(awk "NR==$SLURM_ARRAY_TASK_ID" "$SEEDFILE")
eval "$SEED"
